from matplotlib import pyplot as plt
import Utils
from itertools import zip_longest
import json

with open('Q_learning.json', 'r') as file:
    list1 = json.load(file)

# 读取第二个文件中的列表
with open('Lattice.json', 'r') as file:
    list2 = json.load(file)

with open('Constraint_GA.json', 'r') as file:
    list3 = json.load(file)


# for i in range(len(combined_list)):
#     averages_adrs = list(list(map(lattice_utils.avg, zip_longest(*combined_list[i]))))
#     plt.title("ADRS evolution")
#     plt.ylabel("mean ADRS")
#     plt.xlabel("# of synthesis")
#     plt.plot(range(population_size[i], len(averages_adrs) + population_size[i]), averages_adrs)
#     plt.grid()
#     # plt.show()
#     plt.savefig('All ADRS evolution2.png')


averages_adrs1 = list(list(map(lattice_utils.avg, zip_longest(*list1))))
averages_adrs2 = list(list(map(lattice_utils.avg, zip_longest(*list2))))
averages_adrs3 = list(list(map(lattice_utils.avg, zip_longest(*list3))))
plt.title("ADRS evolution")
plt.ylabel("mean ADRS")
plt.xlabel("# of synthesis")
plt.plot(range(100, len(averages_adrs1) + 100), averages_adrs1,color = "red")
plt.plot(range(100, len(averages_adrs2) + 100), averages_adrs2)
plt.plot(range(100, len(averages_adrs3) + 100), averages_adrs3,color = "green")
plt.grid()
# plt.show()
plt.savefig('All ADRS evolution2.png')