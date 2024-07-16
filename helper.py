import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(values, avg_values):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    plt.plot(values)
    plt.plot(avg_values)
    plt.ylim(ymin=0)
    plt.text(len(values)-1, values[-1], str(values[-1]))
    plt.text(len(avg_values)-1, avg_values[-1], str(avg_values[-1]))
    plt.show()

    