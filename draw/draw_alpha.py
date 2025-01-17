import matplotlib.pyplot as plt

data = [
    1.591346530045176,
    1.4398794558793426,
    1.7547521722414199,
    1.8063566719121205,
    1.3076706063873476,

    2.870853090210214,
    1.303663225387199,
    2.3672818229044883,
    1.7256119060686523,
    1.8976244437398908,

    1.4073054951204143,
    2.36077384162972,
    3.118361686026028,
    1.7245310818425226,
    1.280292102282859,

    1.6250235000793134,
    1.6296912226507223,
    1.7810049628942721,
    2.0706026081259403,
    1.8606337340686823
]

plt.figure(figsize=(10, 6))
plt.bar(range(len(data)), data, color='gray')

plt.axhline(y=3, color='red', linestyle='--', linewidth=2)

plt.ylim(0, 3.5)  

plt.title("The distribution of the alpha values observation in 20 WSIs from TCGA.")
plt.ylabel("Alpha Value")

plt.savefig("draw/pics/sorted_alpha_bar_chart.png")

plt.close()
