from mpmath import mp

mp.dps = 500000  # Set precision to 500,000 decimal places

# Function to insert <br> every couple of characters
def insert_br_every_two_chars(input_str):
    return '<br>'.join(input_str[i:i+2] for i in range(0, len(input_str), 2))

pi_str = str(mp.pi)  # Get the string representation of π
formatted_pi = insert_br_every_two_chars(pi_str)  # Insert <br> tags

output_pi = "<h1>PI</h1><div>" + formatted_pi + "</div>"

with open("index.html", "w") as file:
    file.write(output_pi)
