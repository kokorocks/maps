from mpmath import mp
mp.dps = 500000  # Set precision to 50 decimal places
#print(mp.pi)

output_pi = "<h1>PI</h1><div>"+str(mp.pi)+"</div>"

with open("index.html", "w") as file:
    file.write(output_pi)

