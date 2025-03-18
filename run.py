from mpmath import mp
mp.dps = 50000000  # Set precision to 50 decimal places
print(mp.pi)

with open("output.html", "w") as file:
    file.write('<h1>PI</h1><div>'+mp.pi+'</div>')