def DINOv3StateDictConverter(state_dict):
    new_state_dict = {}
    for key in state_dict:
        value = state_dict[key]
        new_key = key.removeprefix("model.") if key.startswith("model.layer.") else key
        if new_key in new_state_dict:
            raise ValueError(f"DINOv3 state-dict key collision: {key} -> {new_key}")
        new_state_dict[new_key] = value
    return new_state_dict
