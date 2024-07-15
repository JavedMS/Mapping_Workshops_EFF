import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statistics
import matplotlib.ticker as mtick
from matplotlib.patches import Ellipse
from wordcloud import WordCloud
import re
from cartopy import crs as ccrs
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors

border_color = "light grey"
colors = ['#ffa0a0', '#ffb0a0', '#ffc0a0', '#ffd0a0', '#ffe0a0', '#fff0a0']


def line_plot(df: pd.Series, cols: list[str]):
    # Assume that all the columns are in the datastructure 

    return None

def donut_plot(df: pd.Series, fig, ax):
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
    #ax.set_ylabel("Y-axis", rotation = "horizontal")
    #ax.set_xlabel("X-axis", rotation = "horizontal")
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

    #plt.show()

def box_plot(df: pd.DataFrame):
    sorted_columns = df.mean().sort_values(ascending = False).index
    data = pd.melt(df[sorted_columns])

    # Create the box plot
    sns.boxplot(x='Response', y='value', data=data, width = 0.5, color = "lightgray")

    # Enhance the plot
    plt.xticks(rotation=45) # Rotate the x-axis labels for better readability
    plt.show()

def violin_chart(df: pd.DataFrame, cols: list[str]):
    return None

def bar_chart(plot_data: pd.DataFrame, fig, ax):
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
    plt.tight_layout()
    plt.show()

def barh_chart(plot_data: pd.DataFrame, fig, ax):
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [6.4,4.8])

    # Remove all spines
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    # Add the top spine
    ax.spines["top"].set(color = "gray", visible = True)
    ax.spines["top"].set_position(position = ("outward",10))

    #ax.set_xlim(0,5)

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

def text_pre(df: pd.DataFrame, col: str):
    # Creaete a pandas.Series object and split it into words
    split_symbols = "[\n1.2.3.() /,-]"
    symbols = df[col].apply(lambda x: list(filter(None, re.split(split_symbols, x.lower()))))

    def remove_short(x: str):
        if (len(x) > 3):
            return x
        else:
            return None

    symbols = symbols.apply(lambda x: list(filter(remove_short,x)))

    text = symbols.explode().to_list() # Flatten the list  

    # Create wordcloud
    text_str = " ".join([str(t) for t in text]) # Convert to a string object
    return text_str

def word_cloud(df: pd.DataFrame, col: str):
    text_str = text_pre(df, col)
    wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        stopwords = None,  # You can add a set of words to exclude here
                        min_font_size = 10).generate(text_str)

    # Display the generated image:
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    

    plt.show()

def hist_plot(df: pd.DataFrame, ax: mpl.axes):
    sns.histplot(df, discrete=True, element="poly", alpha = 0.1, ax = ax)

    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    ax.grid(visible = True, axis = "y")

def bag_of_words(symbols: pd.Series):
    # Create bag of words representation to find word frequency
    bag_of_words = symbols.str.get_dummies(sep=' ')
    word_counts = bag_of_words.sum(axis=0)
    top_words = word_counts.nlargest(10)
    print(top_words)

def geo_plot(df, fig, ax):
    def to_percent(x, pos):
        return f'{x * 100:.0f}%'

    formatter = FuncFormatter(to_percent)
    df.plot(column = "pos", 
              cmap = 'copper',
              edgecolor = "white", 
              legend = False, 
              legend_kwds=
              {
                  "drawedges": False,
                  "label": "", 
                  "orientation": "vertical",
                  "format": formatter,
                  },
              missing_kwds = # If a municipality is not chosen
              { 
                "color": "lightgrey",
                "edgecolor": "red",
                "hatch": "///",
                "label": "Missing values"
              },
              ax = ax)

    # Remove spines and markers
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    # Remove axis
    ax.set_axis_off()
    ax.set_xlabel("X-axis")

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap='copper', norm=plt.Normalize(vmin=df['pos'].min(), vmax=df['pos'].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.outline.set_visible(False)  # Remove the colorbar frame

    cmap = cbar.cmap # Get the colormap for the colorbar

    vmin, vmax = (df["pos"].min(), df["pos"].max())
    # Normalize object to map the value range to [0, 1]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for label in cbar.ax.get_yticklabels():
        val = float(label.get_text())
        normalized_val = norm(val)
        color = cmap(normalized_val)

        # set function for mpl.Text
        label.set(color = color, fontsize=10) 
        label.set_text(to_percent(val,0)) # Why doesn't this work? 

    # Remove the tick-length
    cbar.ax.tick_params(length=0)

    # Modify the legend
    legend = ax.get_legend()

def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())


    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)

    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax
