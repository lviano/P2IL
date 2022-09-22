import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
 
labels = ['Dual FORB', 
          'Dual SGD', 
          'Primal Dual FORB', 
          'Primal Dual SGD', 
          "AIRL", 
          "AIRL Linear", 
          "GAIL", 
          "GAIL Linear", 
          "IQLearn SGD", 
          "IQLearn FORB"]
colors = ["blue", "orange", 
            "green", "red", 
            "black", "brown", 
            "gray", "darkcyan",
            "yellow", "goldenrod"]
 
fig = plt.figure()
fig_legend = plt.figure(figsize=(2, 3))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[2])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[3])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[4])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[5])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[6])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[7])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[8])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[9])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True)
plt.savefig("figs/legend.pdf")

labels = ['PP FORB', 
          'PP SGD', 
          'PP Adam',
          'Primal Dual FORB', 
          'Primal Dual SGD']
colors = ["blue", "orange", "turquoise",
            "green", "red"]
 
fig = plt.figure()
fig_legend = plt.figure(figsize=(2, 3))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[2])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[3])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[4])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True)
plt.savefig("figs/ablation_legend.pdf")

labels = ['Proximal Point',
          #'Mirror Descent',
          'IQLearn',
          'AIRL',
          'GAIL',
          'AIRL Linear',
          'GAIL Linear']
colors = ["blue", #"green", 
            "goldenrod",
            "black", 
            "gray", "brown", "darkcyan",]
 
fig = plt.figure()
fig_legend = plt.figure(figsize=(2, 3))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[2])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[3])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[4])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[5])[0])
#lines.append(ax.plot(range(2), range(2), '-o', c=colors[6])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True)
plt.savefig("figs/final_paper_legend.pdf")
labels = ['Proximal Point',
          'Mirror Descent']
colors = ["blue", "green"]
fig = plt.figure()
fig_legend = plt.figure(figsize=(2, 3))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
#lines.append(ax.plot(range(2), range(2), '-o', c=colors[6])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True)
plt.savefig("figs/ours_legend.pdf")
labels = ['Proximal Point',
          #'Mirror Descent',
          'IQLearn',
          'AIRL',
          'GAIL',
          'AIRL Linear',
          'GAIL Linear']
colors = ["blue", #"green", 
            "goldenrod",
            "black", 
            "gray", "brown", "darkcyan",]
fig = plt.figure()
fig_legend = plt.figure(figsize=(10, 1))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[2])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[3])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[4])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[5])[0])
#lines.append(ax.plot(range(2), range(2), '-o', c=colors[6])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True, ncol=6)
plt.savefig("figs/final_paper_legend_horizontal.pdf")

labels = ['Proximal Point',
          #'Mirror Descent',
          'IQLearn',
          'AVRIL']
colors = ["blue", #"green", 
            "goldenrod",
            "green"]
fig = plt.figure()
fig_legend = plt.figure(figsize=(10, 1))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[2])[0])
#lines.append(ax.plot(range(2), range(2), '-o', c=colors[6])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True, ncol=6)
plt.savefig("figs/final_paper_legend_horizontal_avril.pdf")

labels = ['Proximal Point',
          #'Mirror Descent',
          'IQLearn',
          'AIRL',
          'GAIL']
colors = ["blue", #"green", 
            "goldenrod",
            "black", 
            "gray"]
fig = plt.figure()
fig_legend = plt.figure(figsize=(10, 1))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[2])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[3])[0])
#lines.append(ax.plot(range(2), range(2), '-o', c=colors[6])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True, ncol=4)
plt.savefig("figs/legend_no_linear.pdf")


labels = ['Proximal Point',
          #'Mirror Descent',
          'IQLearn',
          'AVRIL']
colors = ["blue", #"green", 
            "goldenrod",
            "green"]
 
fig = plt.figure()
fig_legend = plt.figure(figsize=(2, 1))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[2])[0])
#lines.append(ax.plot(range(2), range(2), '-o', c=colors[6])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True)
plt.savefig("figs/offline_legend.pdf")

labels = ['Proximal Point',
          #'Mirror Descent',
          'IQLearn',
          'AIRL',
          'GAIL']
colors = ["blue", #"green", 
            "goldenrod",
            "black", 
            "gray"]
 
fig = plt.figure()
fig_legend = plt.figure(figsize=(2, 1))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[2])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[3])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True)
plt.savefig("figs/atari_legend.pdf")