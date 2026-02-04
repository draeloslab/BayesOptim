import pandas as pd

infile = 'profile_output.txt'
outfile = 'profile_output.csv'

rows = []
columns = ['Line #', 'Hits', 'Time', 'Per Hit', '% Time', 'Line Contents', 'Function']

with open(infile, 'r') as f:
    in_block = False
    function = ''
    for line in f:
        if line.split(' ')[0] == 'Function:':
            function = line.split(' ')[1]
        elif line[0] == '=': # line of only '=' indicates start of table block
            in_block = True
            continue
        elif in_block and not line.strip(): # empty line indicates end of table block
            in_block = False
            continue
        elif in_block:
            parts = line.strip().split(maxsplit=5)
            if len(parts) == 1: continue # skip empty lines
            elif not parts[1].isnumeric(): continue # skip untimed lines
            parts.append(function)
            rows.append(parts)

# Create dataframe, change types to numeric
df = pd.DataFrame(rows, columns=columns)
df['Line #'] = df['Line #'].astype(int)
df['Hits'] = df['Hits'].astype(int)
df['Time'] = df['Time'].astype(float)
df['Per Hit'] = df['Per Hit'].astype(float)
df['% Time'] = df['% Time'].astype(float)

# Determine which lines are significant, output to csv
df = df[df['Time'] >= 10000000] # at least 10 ms total
df.to_csv(outfile, index=False)
print(f"Output to {outfile}")
