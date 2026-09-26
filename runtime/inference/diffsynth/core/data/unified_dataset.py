from .operators import *
import hashlib
import os
import torch, json, pandas


PHY_PARAM_CONDITION_CONTRACT = "phyparam-canonical-fmueg-v1"
PHY_PARAM_DINO_SHA256 = "f286f96b8e284894a88f97b0243d9ed6d9b3ace6383af32e28398eb39d9c5c85"
PHY_PARAM_CONDITION_KEYS = {
    "first_frame_masks", "object_valid_mask", "force", "force_present",
    "gravity", "gravity_present", "mu", "mu_present", "restitution",
    "restitution_present", "mass", "mass_present",
    "phyparam_condition_contract", "phyparam_identity",
}
PHY_PARAM_REQUIRED_CONDITION_KEYS = PHY_PARAM_CONDITION_KEYS - {"mass", "mass_present"}


def _validate_phyparam_bundle(condition, dino, expected):
    if not isinstance(condition, dict) or not isinstance(dino, dict):
        raise TypeError("PhyParam cache bundle must contain two mappings")
    keys = set(condition)
    if not PHY_PARAM_REQUIRED_CONDITION_KEYS <= keys or keys - PHY_PARAM_CONDITION_KEYS:
        raise ValueError("PhyParam condition schema differs from the frozen contract")
    if condition["phyparam_condition_contract"] != PHY_PARAM_CONDITION_CONTRACT:
        raise ValueError("PhyParam condition contract differs from the frozen contract")
    expected_identity = {
        key: str(expected[key])
        for key in (
            "source", "raw_clip_id", "clip_id", "video", "video_sha256",
            "source_row_sha256",
        )
    }
    if condition["phyparam_identity"] != expected_identity:
        raise ValueError(f"PhyParam condition identity differs: {expected_identity['clip_id']}")
    if (
        dino.get("source_video") != expected_identity["video"]
        or dino.get("source_video_sha256") != expected_identity["video_sha256"]
        or dino.get("feature_contract") != "final_norm_patch_tokens_without_cls_or_registers"
        or dino.get("teacher_model_sha256") != PHY_PARAM_DINO_SHA256
        or dino.get("decoded_frames") != 49
        or dino.get("feature_frame_indices") != [0,4,8,12,16,20,24,28,32,36,40,44,48]
        or dino.get("training_spatial_preprocess") != {"height":448,"width":768,"resize_mode":"pad"}
        or tuple(dino.get("spatial_shape", ())) != (14, 14)
    ):
        raise ValueError(f"PhyParam DINO identity differs: {expected_identity['clip_id']}")
    features = dino.get("features")
    if (
        not torch.is_tensor(features)
        or tuple(features.shape) != (13, 14, 14, 4096)
        or features.dtype != torch.bfloat16
        or not torch.isfinite(features).all()
    ):
        raise ValueError(f"PhyParam DINO tensor contract differs: {expected_identity['clip_id']}")


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
        cache_override_manifest=None,
        cache_override_key=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.cache_override_key = cache_override_key
        self.cache_overrides = {}
        self.cache_override_aux = {}
        self.cache_prompt_overrides = {}
        self.cache_override_identity = {}
        strict_hashes = os.environ.get(
            "PHYSICAL_WM_STRICT_CACHE_OVERRIDE_HASHES", "0"
        ).strip().lower()
        if strict_hashes not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("PHYSICAL_WM_STRICT_CACHE_OVERRIDE_HASHES must be boolean")
        self.strict_cache_override_hashes = strict_hashes in {
            "1", "true", "yes", "on"
        }
        self.cache_override_sha256 = {}
        self.cache_source_sha256 = {}
        self.cache_override_mtime_ns = {}
        self.cache_source_mtime_ns = {}
        self.verified_cache_overrides = set()
        self.verified_cache_sources = set()
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
        self.load_cache_overrides(cache_override_manifest)

    def load_cache_overrides(self, manifest_path):
        if manifest_path is None:
            if self.cache_override_key is not None:
                raise ValueError("cache_override_key requires cache_override_manifest")
            return
        if not self.load_from_cache:
            raise ValueError("cache overrides are valid only with preencoded caches")
        if self.cache_override_key not in {
            "sparse_object_condition",
            "sparse_object_condition_with_prompt",
            "unified_control_latents",
            "phyparam_bundle",
        }:
            raise ValueError(f"unsupported cache override key: {self.cache_override_key!r}")
        rows = pandas.read_csv(manifest_path).to_dict("records")
        required = {
            "cache_file",
            "override_path",
            "cache_source",
            "cache_source_bytes",
        }
        if self.strict_cache_override_hashes:
            required |= {
                "cache_source_sha256",
                "cache_source_mtime_ns",
                "override_sha256",
                "override_mtime_ns",
            }
        ordered_cache_files = []
        override_rows = {}
        for row in rows:
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"cache override manifest missing columns {sorted(missing)}"
                )
            cache_file = str(row["cache_file"])
            if os.path.basename(cache_file) != cache_file or not cache_file.endswith(".pth"):
                raise ValueError(f"cache_file must be a .pth basename: {cache_file!r}")
            if cache_file in self.cache_overrides:
                raise ValueError(f"duplicate cache override row: {cache_file}")
            override_path = os.path.abspath(str(row["override_path"]))
            if not os.path.isfile(override_path):
                raise FileNotFoundError(override_path)
            self.cache_overrides[cache_file] = override_path
            if self.strict_cache_override_hashes:
                cache_sha256 = str(row["cache_source_sha256"])
                override_sha256 = str(row["override_sha256"])
                if len(cache_sha256) != 64 or len(override_sha256) != 64:
                    raise ValueError(f"content sha256 differs for {cache_file}")
                self.cache_source_sha256[cache_file] = cache_sha256
                self.cache_override_sha256[cache_file] = override_sha256
                self.cache_override_mtime_ns[cache_file] = int(
                    row["override_mtime_ns"]
                )
                self.cache_source_mtime_ns[cache_file] = int(
                    row["cache_source_mtime_ns"]
                )
            if self.cache_override_key == "phyparam_bundle":
                dino_path = os.path.abspath(str(row.get("phyparam_dino_features_path", "")))
                if not os.path.isfile(dino_path):
                    raise FileNotFoundError(dino_path)
                self.cache_override_aux[cache_file] = dino_path
                identity_fields = {
                    "source", "raw_clip_id", "clip_id", "video", "video_sha256",
                    "source_row_sha256", "phyparam_condition_contract",
                }
                missing_identity = identity_fields - set(row)
                if missing_identity:
                    raise ValueError(
                        f"PhyParam bundle row lacks identity fields {sorted(missing_identity)}"
                    )
                if row["phyparam_condition_contract"] != PHY_PARAM_CONDITION_CONTRACT:
                    raise ValueError("PhyParam override manifest contract differs")
                self.cache_override_identity[cache_file] = {
                    key: str(row[key]) for key in identity_fields
                }
            elif self.cache_override_key == "sparse_object_condition_with_prompt":
                prompt_path = row.get("prompt_context_path", "")
                if isinstance(prompt_path, str) and prompt_path.strip():
                    prompt_path = os.path.abspath(prompt_path)
                    if not os.path.isfile(prompt_path):
                        raise FileNotFoundError(prompt_path)
                    self.cache_prompt_overrides[cache_file] = prompt_path
            ordered_cache_files.append(cache_file)
            override_rows[cache_file] = row
        cache_files = [os.path.basename(path) for path in self.cached_data]
        if len(cache_files) != len(set(cache_files)):
            raise ValueError("cached .pth basenames must be unique when overrides are used")
        cache_by_file = {
            os.path.basename(path): path for path in self.cached_data
        }
        missing = set(cache_files) - set(self.cache_overrides)
        extra = set(self.cache_overrides) - set(cache_files)
        if missing or extra:
            raise ValueError(
                "cache override mapping must be one-to-one: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        ordered_cache_paths = []
        for cache_file in ordered_cache_files:
            cache_path = cache_by_file[cache_file]
            row = override_rows[cache_file]
            if os.path.realpath(cache_path) != os.path.realpath(
                str(row["cache_source"])
            ):
                raise ValueError(f"cache source differs for {cache_file}")
            if os.path.getsize(cache_path) != int(row["cache_source_bytes"]):
                raise ValueError(f"cache size differs for {cache_file}")
            if (
                self.strict_cache_override_hashes
                and os.stat(cache_path).st_mtime_ns
                != self.cache_source_mtime_ns[cache_file]
            ):
                raise ValueError(f"cache mtime differs for {cache_file}")
            ordered_cache_paths.append(cache_path)
        # The signed override manifest is the authoritative dataset order.
        # Every distributed rank therefore exposes the same sampler index.
        self.cached_data = ordered_cache_paths
        index_digest = hashlib.sha256(
            "\n".join(ordered_cache_files).encode("utf-8")
        ).hexdigest()
        print(f"cache index sha256={index_digest}")
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        frame_rate=24, fix_frame_rate=False,
        resize_mode="crop",
    ):
        frame_processor = ImageCropAndResize(
            height,
            width,
            max_pixels,
            height_division_factor,
            width_division_factor,
            resize_mode=resize_mode,
        )
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp", "bmp"), LoadImage() >> frame_processor >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=frame_processor,
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=frame_processor,
                    frame_rate=frame_rate, fix_frame_rate=fix_frame_rate,
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        # Every distributed rank must construct the same cache index before
        # the sampler shards it.  Filesystem enumeration order is not a stable
        # cross-node contract on PFS.
        for file_name in sorted(os.listdir(path)):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            cache_path = self.cached_data[data_id % len(self.cached_data)]
            cache_file = os.path.basename(cache_path)
            if self.strict_cache_override_hashes:
                if (
                    os.stat(cache_path).st_mtime_ns
                    != self.cache_source_mtime_ns[cache_file]
                ):
                    raise ValueError(f"cache mtime differs for {cache_file}")
                if cache_file not in self.verified_cache_sources:
                    digest = hashlib.sha256()
                    with open(cache_path, "rb") as handle:
                        for chunk in iter(lambda: handle.read(8 << 20), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != self.cache_source_sha256[cache_file]:
                        raise ValueError(f"cache sha256 differs for {cache_file}")
                    self.verified_cache_sources.add(cache_file)
            data = self.cached_data_operator(cache_path)
            if self.cache_overrides:
                if not (
                    isinstance(data, (tuple, list))
                    and len(data) == 3
                    and isinstance(data[0], dict)
                ):
                    raise TypeError("preencoded Wan cache must be a 3-item tuple")
                shared = dict(data[0])
                shared.pop("sparse_object_condition", None)
                shared.pop("unified_control_latents", None)
                shared.pop("unified_control_present", None)
                shared.pop("phyparam_dino_features", None)
                override_path = self.cache_overrides[cache_file]
                if self.strict_cache_override_hashes:
                    if (
                        os.stat(override_path).st_mtime_ns
                        != self.cache_override_mtime_ns[cache_file]
                    ):
                        raise ValueError(
                            f"condition mtime differs for {cache_file}"
                        )
                    if cache_file not in self.verified_cache_overrides:
                        digest = hashlib.sha256()
                        with open(override_path, "rb") as handle:
                            for chunk in iter(lambda: handle.read(8 << 20), b""):
                                digest.update(chunk)
                        if digest.hexdigest() != self.cache_override_sha256[cache_file]:
                            raise ValueError(
                                f"condition sha256 differs for {cache_file}"
                            )
                        self.verified_cache_overrides.add(cache_file)
                if self.cache_override_key == "phyparam_bundle":
                    condition = self.cached_data_operator(override_path)
                    dino = self.cached_data_operator(self.cache_override_aux[cache_file])
                    _validate_phyparam_bundle(
                        condition, dino, self.cache_override_identity[cache_file]
                    )
                    shared["sparse_object_condition"] = condition
                    shared["phyparam_dino_features"] = dino
                    positive = data[1]
                elif self.cache_override_key == "sparse_object_condition_with_prompt":
                    shared["sparse_object_condition"] = self.cached_data_operator(
                        override_path
                    )
                    positive = data[1]
                    prompt_path = self.cache_prompt_overrides.get(cache_file)
                    if prompt_path is not None:
                        positive = self.cached_data_operator(prompt_path)
                        if (
                            not isinstance(positive, dict)
                            or set(positive) != {"prompt", "context"}
                            or not isinstance(positive["prompt"], str)
                            or not torch.is_tensor(positive["context"])
                            or tuple(positive["context"].shape) != (1, 512, 4096)
                        ):
                            raise ValueError(
                                f"prompt context override differs: {prompt_path}"
                            )
                else:
                    shared[self.cache_override_key] = self.cached_data_operator(override_path)
                    positive = data[1]
                if self.cache_override_key == "unified_control_latents":
                    shared["unified_control_present"] = True
                data = (shared, positive, data[2])
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
