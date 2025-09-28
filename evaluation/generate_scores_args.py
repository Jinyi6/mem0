import json
import pandas as pd
import argparse

def calculate_and_save_scores(input_file, output_file):
    """
    Loads evaluation metrics, calculates aggregate scores, and saves them to a text file.
    """
    # Load the evaluation metrics data
    with open(input_file, "r") as f:
        data = json.load(f)

    # Flatten the data into a list of question items
    all_items = []
    for key in data:
        all_items.extend(data[key])
    
    if not all_items:
        print("Warning: No items found in the input file to evaluate.")
        with open(output_file, 'w') as f:
            f.write("No data to evaluate.\n")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_items)

    # Convert columns to numeric, coercing errors
    for col in ["category", "bleu_score", "f1_score", "llm_score"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where conversion failed
    df.dropna(subset=["category", "bleu_score", "f1_score", "llm_score"], inplace=True)

    # Calculate mean scores by category
    result = df.groupby("category").agg({
        "bleu_score": "mean", 
        "f1_score": "mean", 
        "llm_score": "mean"
    }).round(4)

    # Add count of questions per category
    result["count"] = df.groupby("category").size()

    # Calculate overall means
    overall_means = df[["bleu_score", "f1_score", "llm_score"]].mean().round(4)

    # Save the results to the output file
    with open(output_file, "w") as f:
        f.write("Mean Scores Per Category:\n")
        f.write(result.to_string())
        f.write("\n\n" + "="*30 + "\n\n")
        f.write("Overall Mean Scores:\n")
        f.write(overall_means.to_string())

    print(f"Final scores calculated and saved to {output_file}")
    print("\nMean Scores Per Category:")
    print(result)
    print("\nOverall Mean Scores:")
    print(overall_means)

def main():
    parser = argparse.ArgumentParser(description="Calculate and save evaluation scores.")
    parser.add_argument("--input_file", required=True, help="Path to the input evaluation_metrics.json file.")
    parser.add_argument("--output_file", required=True, help="Path to save the final scores text file.")
    
    args = parser.parse_args()
    
    calculate_and_save_scores(args.input_file, args.output_file)

if __name__ == "__main__":
    main()