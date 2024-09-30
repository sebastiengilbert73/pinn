import logging
import os
import argparse
import pandas as pd
import ast
import statistics

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    baseDirectory,
    outputDirectory,
    epochLossFilename,
    columnNames
):
    logging.info("extract_experiment_results.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Get the directories under the base directory
    subdirectories = directories_under(baseDirectory)

    columnName_to_values = {col: [] for col in columnNames}
    for subdirectory in subdirectories:
        epoch_loss_filepath = os.path.join(subdirectory, epochLossFilename)
        if os.path.exists(epoch_loss_filepath):
            epoch_loss_df = pd.read_csv(epoch_loss_filepath)
            for col in columnNames:
                values = epoch_loss_df[col].tolist()
                columnName_to_values[col].append(values)

    number_of_epochs = 0
    columnName_to_epochMeanStdev = {col: [] for col in columnNames}
    for column_name, values_list_list in columnName_to_values.items():
        number_of_runs = len(values_list_list)
        number_of_epochs = len(values_list_list[0])
        #logging.info(f"column_name = {column_name}; number_of_runs = {number_of_runs}; number_of_epochs = {number_of_epochs}")
        for epoch in range(number_of_epochs):
            run_values = [values_list_list[run][epoch] for run in range(number_of_runs)]
            #for run in number_of_runs:
            #    run_values.append(values_list_list[run][epoch])
            run_mean = statistics.mean(run_values)
            run_std_dev = statistics.stdev(run_values)
            columnName_to_epochMeanStdev[column_name].append((run_mean, run_std_dev))

    # Write the statistics to a file
    with open(os.path.join(outputDirectory, "results_summary.csv"), 'w') as results_summary_file:
        results_summary_file.write(f"epoch")
        for col in columnNames:
            results_summary_file.write(f",{col}_mean,{col}_stdDev")
        results_summary_file.write("\n")
        for epoch in range(len(range(number_of_epochs))):
            results_summary_file.write(f"{epoch + 1}")
            for col in columnNames:
                results_summary_file.write(f",{columnName_to_epochMeanStdev[col][epoch][0]},{columnName_to_epochMeanStdev[col][epoch][1]}")
            results_summary_file.write("\n")

def directories_under(base_directory):
    subdirectories = []
    for root, dirs, files in os.walk(base_directory):
        for directory in dirs:
            subdirectories.append(os.path.join(root, directory))
    return subdirectories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseDirectory', help="The directory that contains the directories containing the epochLoss.csv files")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_extract_experiment_results'",
                        default='./output_extract_experiment_results')
    parser.add_argument('--epochLossFilename', help="The filename of the results csv file. Default: 'epochLoss.csv'", default='epochLoss.csv')
    parser.add_argument('--columnNames', help="The column names. Default: ['loss', 'initial_loss', 'boundary_loss', 'diff_eqn_loss']", default="['loss', 'initial_loss', 'boundary_loss', 'diff_eqn_loss']")
    args = parser.parse_args()
    args.columnNames = ast.literal_eval(args.columnNames)
    main(
        args.baseDirectory,
        args.outputDirectory,
        args.epochLossFilename,
        args.columnNames
    )