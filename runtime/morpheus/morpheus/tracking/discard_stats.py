import os
import argparse
import json
import datetime
from .processing import load_centers3d_over_time
from .discard_filtering import filter_trajectory
import json


def calculate_permanence_stats(base_path: str, recompute_filtering_info: bool = False):
    """
    Calculate and save object permanence statistics for a batch of videos.
    
    Args:
        base_path: Directory containing the tracking log files
    """
    total_videos = 0
    short_invalid_trajectories_initial = 0
    discarded_short_trajectories = 0
    shortned_valid_trajectories = 0

    print(f"Calculating object permanence stats for {base_path}")
    
    # Collect all tracking log files
    json_files = []
    for root, _, files in os.walk(base_path):
        if files and "masks.pkl" in files:
            should_compute = recompute_filtering_info or "filtering_info.json" not in files
            if should_compute:
                action = "Recomputing" if recompute_filtering_info else "Building"
                print(f"{action} filtering_info.json files for {root} from the .pkl files")
                centers3d_over_time = load_centers3d_over_time(root)
                discard_details = filter_trajectory(centers3d_over_time, gap_threshold=0.15, min_trajectory_length=0.15, min_valid_percentage=0.15)
                
                filtering_info = {
                    "trajectory_filtering": {
                        "discard_details": discard_details,
                        "gap_threshold": 0.15,
                        "min_trajectory_length": 0.15,
                        "min_valid_percentage": 0.15
                    }
                }
                
                json_path = os.path.join(root, "filtering_info.json")
                json_files.append(json_path)
                with open(json_path, 'w') as f:
                    json.dump(filtering_info, f, indent=2)

            else:
                print(f"Using existing filtering_info.json files for {root}")
                json_files.append(os.path.join(root, "filtering_info.json"))

    total_videos = len(json_files)
    if total_videos == 0:
        print("No filtering_info.json files found!")
        return
    
    # Process each json file
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        min_trajectory_length = data["trajectory_filtering"]["min_trajectory_length"]
        min_valid_percentage = data["trajectory_filtering"]["min_valid_percentage"]
        for object_id, discard_details in data["trajectory_filtering"]["discard_details"].items():
            if discard_details["is_discarded"]:
                if "None frames too many smaller than min. valid_threshold" in discard_details["discard_reason"]:
                    short_invalid_trajectories_initial += 1
                    break
                elif "Longest segment of object detection too short" in discard_details["discard_reason"]:
                    discarded_short_trajectories += 1
                    break
            if discard_details["shortened_trajectory"]:
                shortned_valid_trajectories += 1
                break
    # Calculate percentages
    videos_filtered_completely = short_invalid_trajectories_initial + discarded_short_trajectories
    percent_filtered_completely = (videos_filtered_completely / total_videos) * 100
    percent_shortened_valid_trajectories = (shortned_valid_trajectories / total_videos) * 100
    # Create statistics dictionary
    stats = {
        'total_videos': total_videos,
        'videos_filtered_completely': videos_filtered_completely,
        'percent_filtered_completely': percent_filtered_completely,
        'short_invalid_trajectories_initial': short_invalid_trajectories_initial,
        'discarded_shortned_trajectories': discarded_short_trajectories,
        'shortened_valid_trajectories': shortned_valid_trajectories,
        'percent_shortened_valid_trajectories': percent_shortened_valid_trajectories,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save JSON format
    json_file = os.path.join(base_path, 'permanence_stats.json')
    with open(json_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Save TXT format
    txt_file = os.path.join(base_path, 'permanence_stats.txt')
    with open(txt_file, 'w') as f:
        f.write("Object Permanence Statistics\n")
        f.write("===========================\n\n")
        f.write(f"Generated on: {stats['timestamp']}\n\n")
        f.write(f"Total videos analyzed: {stats['total_videos']}\n")
        f.write(f"Videos filtered completely: {stats['videos_filtered_completely']} ")
        f.write(f"({stats['percent_filtered_completely']:.1f}%)\n")
        f.write(f"Videos with shortened trajectories: {stats['shortened_valid_trajectories']} ")
        f.write(f"({stats['percent_shortened_valid_trajectories']:.1f}%)\n")
        f.write("\nBreakdown of filtered videos:\n")
        f.write(f"- Initial invalid short trajectories (Containing None frames above threshold: {min_valid_percentage}): {stats['short_invalid_trajectories_initial']}\n")
        f.write(f"- Discarded shortned trajectories ( due Longest segment of object detection smaller than threshold: {min_trajectory_length}): {stats['discarded_shortned_trajectories']}\n")
    
    print(f"\nStatistics saved to:\n{json_file}\n{txt_file}")
    
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate object permanence statistics from tracking logs')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing tracking log files')
    
    args = parser.parse_args()
    
    # If output file not specified, create one based on experiment and model names

    stats = calculate_permanence_stats(args.base_dir) 