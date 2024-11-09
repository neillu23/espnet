import pandas as pd



asr = pd.read_csv('/projects/p32222/ylu125/espnet_outeff/egs2/voxtlm_v1/lm1/dump/raw/test/text.asr', delimiter=' ', header=None)
print(asr.head(2))
def process_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        parts = line.split(' ', 1)  # Split only at the first space
        if len(parts) == 2:  # Ensure there are two parts after the split
            data.append([parts[0], parts[1].strip()])

    # Create DataFrame
    df = pd.DataFrame(data, columns=[0,1])
    return df

# Example usage
file_path = '/projects/p32222/ylu125/espnet_outeff/egs2/voxtlm_v1/lm1/dump/raw/test/speech/asr/text'
expect_output = process_text_file(file_path)



for i in range(len(asr)):
    # Find matching rows in expect_output where column 0 matches
    index = asr.iloc[i][0].split('_', 1)[1] == expect_output[0]

    # If a match is found, append the text from expect_output
    if index.any():
        matching_row = expect_output[index].iloc[0]  # Get the first matching row
        asr.iloc[i][1] = asr.iloc[i][1] + matching_row[1]  # Append the text from expect_output's column 1 to asr's column 1


asr.to_csv('/projects/p32222/ylu125/espnet_outeff/egs2/voxtlm_v1/lm1/dump/raw/test/text.asr', sep=' ',index=False)