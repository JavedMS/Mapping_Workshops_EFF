import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import matplotlib.ticker as mtick
from matplotlib.patches import Ellipse

border_color = "light grey"
colors = ['#ffa0a0', '#ffb0a0', '#ffc0a0', '#ffd0a0', '#ffe0a0', '#fff0a0']


def line_plot(df: pd.Series, cols: list[str]):
    # Assume that all the columns are in the datastructure 

    return None

def donut_plot(df: pd.Series):
    fig, ax = plt.subplots(nrows=1,ncols = 1)

    percent_distance = 1.3 # Just outside the plot
    explode = [0.01 for _ in range(len(df))] # TODO: Has to be the same size as x 
    angle = 90
    patches, texts, autotexts = ax.pie(df,
            autopct='%1.0f%%', 
            pctdistance= percent_distance, 
            explode = explode, 
            labels = None, 
            startangle=angle,
            colors = colors)

    # Set labels
    ax.set_ylabel("Y-axis", rotation = "horizontal")
    ax.set_xlabel("X-axis", rotation = "horizontal")
    ax.xaxis.set_label_position(position='top')

    # Change text-color
    for i, prc_text in enumerate(autotexts):
        prc_text.set_color(colors[i])
        prc_text.set_fontsize(13)

    # Add the ellipsis in the middle
    width = 0.55 * 2
    centre_ellipsis = Ellipse((0, 0), width, width + explode[0], fc='white', angle = angle)
    fig = plt.gcf()
    fig.gca().add_artist(centre_ellipsis)

    # Add labels
    ax.legend(loc='upper right', 
              labels = df.index,
              labelcolor="gray", 
              facecolor="white", 
              edgecolor="white", 
              frameon=False, 
              bbox_to_anchor=(1.3, 1))

    plt.show()

def violin_chart(df: pd.DataFrame, cols: list[str]):
    return None

def bar_chart(plot_data: pd.DataFrame, cols: list[str]):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [6.4,4.8])

    # Remove all spines
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_visible(True)

    # Add the dataplot with zorder 2 (to be on top)
    bar_width = 0.6
    plot_data.plot(kind='bar', ax=ax, colormap='copper', legend=False, zorder=2, width = bar_width)

    # Ensure the highest point shown is above the highest point
    max_value = (plot_data).max().max()  # Find the maximum value in the dataset after multiplication
    ax.set_ylim(0, max_value * 1.2)  # Increase the max limit by 10% for some padding

    # Change y-axis
    ax.spines['left'].set_position(position=('outward', 30))
    ax.tick_params(axis='y', colors= 'gray')

    # Change bar-colors and the padding between the bars
    groups = statistics.median(range(len(plot_data.columns)))
    j, i = 0, 0
    offset = bar_width * 0.03
    for bar in ax.patches:
        # Adjust the padding calculation to account for the direction
        padding = -offset if i < groups else offset if i > groups else 0
        bar.set(facecolor = colors[i], x = bar.get_x() + padding,  edgecolor = "black")
        j += 1
        i = int(j / len(plot_data.index))

    # Change bar-text
    ax.tick_params(axis='x', length=0, labelrotation=0, labelcolor= "black", pad=15)

    # Change y-acis to percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    # Change legend
    legend_values = plot_data.columns
    ax.legend(legend_values, loc='upper right', labelcolor="gray", facecolor="white", edgecolor="white", frameon=False, bbox_to_anchor=(1, 1.2))

    # Add vertical box-lines with zorder 1 (behind the bars)
    ax.yaxis.grid(visible=True, linestyle="dotted", zorder=1)
    ax.set_axisbelow(True)

    # Remove x_label
    ax.set_xlabel(None)

    # Add title text
    ax.set_title("Wind and Solar Production Over Years ", color = "black", pad = 25, loc = 'left')
    plt.tight_layout()
    plt.show()

def barh_chart(plot_data: pd.DataFrame):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [6.4,4.8])

    # Remove all spines
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    # Add the top spine
    ax.spines["top"].set(color = "gray", visible = True)
    ax.spines["top"].set_position(position = ("outward",10))

    ax.set_xlim(0,5)

    # Add the dataplot with zorder 2 (to be on top)
    bar_width = 0.7
    plot_data.plot(kind='barh', ax=ax, colormap='copper', legend=False, zorder=2, width = bar_width)

    # Change y-axis
    ax.spines['left'].set_position(position=('outward', 10))
    ax.tick_params(axis='y', colors= 'gray')

    # Change bar-colors and the padding between the bars
    i = 0
    for bar in ax.patches:
        # Adjust the padding calculation to account for the direction
        bar.set(facecolor = colors[i],  edgecolor = "black")

    # Change bar-text
    ax.tick_params(axis='x', labelrotation=0, labelcolor= "black", pad=15)

    # Set the x-ticks on top
    ax.tick_params(axis = "x", which = 'both', colors = "gray", top = True, bottom = False, labelbottom = False, labeltop = True)
    ax.tick_params(axis = "y", color = "gray", left = False)

    # Remove x_label
    #ax.set_xlabel(xlabel = "Hello")
    ax.set_xlabel(xlabel = "Respondents where asked to classify from 1 to 5", color = 'gray', loc = 'left', labelpad = 10, fontsize = 7, fontfamily = 'sans-serif')
    ax.xaxis.set_label_position(position='top')
    ax.set_ylabel(None)

    # Add title text
    #ax.set_title("Wind and Solar Production Over Years ", color = "black", pad = 25, loc = 'left')

    plt.tight_layout()
    plt.show()

def word_cloud(df: pd.DataFrame, col: str):
    return None