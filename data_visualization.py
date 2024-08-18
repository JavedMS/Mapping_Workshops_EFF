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
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import textwrap

border_color = "light grey"
colors = ['#ffa0a0', '#ffb0a0', '#ffc0a0', '#ffd0a0', '#ffe0a0', '#fff0a0']

# Seaborn styling
#sns.set_palette('muted')
sns.set_palette('pastel')

# Despine
border_color = "black"
label_color = "black"

# Specifying custom matplotlib.RcParams
rc_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left" : False,
                 "axes.edgecolor" : border_color, "axes.labelcolor" : border_color, 
                 "xtick.color" : border_color, "ytick.color" : border_color, 
                 "xtick.labelcolor" : label_color, "ytick.labelcolor" : label_color,
                 #"axes.grid": True, "grid.color": border_color, "axes.grid.axis": "y"
                 }
sns.set_theme(style="ticks", rc=rc_params)

def line_plot(df: pd.Series, cols: list[str]):
    # Assume that all the columns are in the datastructure 

    return None

def donut_plot(df: pd.Series, fig, ax):
    percent_distance = 1.3 # Just outside the plot
    explode = [0.01 for _ in range(len(df))] # TODO: Has to be the same size as x 
    angle = 90
    color_i = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    patches, texts, autotexts = ax.pie(df,
            autopct='%1.0f%%', 
            pctdistance= percent_distance, 
            explode = explode, 
            labels = None, 
            startangle=angle,
            colors = color_i
            )

    # Set labels
    #ax.set_ylabel("Y-axis", rotation = "horizontal")
    #ax.set_xlabel("X-axis", rotation = "horizontal")
    ax.xaxis.set_label_position(position='top')

    # Change text-color
    for i, prc_text in enumerate(autotexts):
        prc_text.set_color(color_i[i])
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
              bbox_to_anchor=(1.5, 1))

    #plt.show()

def box_plot(df: pd.DataFrame):
    sorted_columns = df.mean().sort_values(ascending = False).index
    data = pd.melt(df[sorted_columns])

    # Create the box plot
    plt.figure(figsize=(18, 6))
    ax = sns.boxplot(x='Response', y='value', data=data, width = 0.6)

    ax.set(xlabel = "", ylabel = "")
    #ax.spines["bottom"].set_visible(False)
    
    ax.tick_params(axis='x', labelsize=17)  # Adjust fontsize as needed
    ax.tick_params(axis='y', labelsize=15)  # Adjust fontsize as needed

    labels = [label.get_text().replace(' ', '\n') for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

    return ax

def violin_chart(df: pd.DataFrame, cols: list[str]):
    return None

def stacked_bar_chart(plot_data: pd.DataFrame, fig, ax):
    # Remove all spines
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_visible(True)

    # Add the dataplot with zorder 2 (to be on top)
    bar_width = 0.6
    plot_data.plot(kind='bar', ax=ax, legend=False, zorder=2, width = bar_width, stacked=True)

    # Change y-axis
    ax.spines['left'].set_position(position=('outward', 30))
    ax.tick_params(axis='y', colors= 'gray')

    # Change bar-text
    ax.tick_params(axis='x', length=0, labelrotation=0, labelcolor= "black", pad=15)

    # Change legend
    legend_values = plot_data.columns
    ax.legend(legend_values, loc='upper right', labelcolor="gray", facecolor="white", edgecolor="white", frameon=False, bbox_to_anchor=(1, 1.2))

    # Add vertical box-lines with zorder 1 (behind the bars)
    ax.yaxis.grid(visible=True, linestyle="dotted", zorder=1)
    ax.set_axisbelow(True)

    # Remove x_label
    ax.set_xlabel(None)

    # Add title text
    #plt.tight_layout()
    plt.show()


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
        bar.set(facecolor = colors[i], x = bar.get_x() + padding)
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
    ax.spines["top"].set(visible = True)
    ax.spines["top"].set_position(position = ("outward",10))

    #ax.set_xlim(0,5)

    # Add the dataplot with zorder 2 (to be on top)
    bar_width = 0.7
    plot_data.plot(kind='barh', ax=ax, legend=False, zorder=2, width = bar_width)

    # Limit the horizontal length of text
    wrapped_labels = ['\n'.join(textwrap.wrap(label.get_text(), width=22)) for label in ax.get_yticklabels()]
    ax.set_yticklabels(wrapped_labels)

    # Change y-axis
    ax.spines['left'].set_position(position=('outward', 10))
    ax.tick_params(axis='y')
    
    # Left align the labels
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('left')

    ax.spines['left'].set_position(position=('outward', 10))

    # Change bar-colors and the padding between the bars
    """
    i = 0
    for bar in ax.patches:
        # Adjust the padding calculation to account for the direction
        bar.set(facecolor = 'tab:blue', edgecolor = "black")
    """
    # Change bar-text
    ax.tick_params(axis='x', labelrotation=0, labelcolor= "black", pad=15)

    # Set the x-ticks on top
    ax.tick_params(axis = "x", which = 'both', top = True, bottom = False, labelbottom = False, labeltop = True)
    ax.tick_params(axis = "y", left = False)

    # Remove x_label
    #ax.set_xlabel(xlabel = "Hello")
    #ax.set_xlabel(xlabel = "Respondents where asked to classify from 1 to 5", color = 'gray', loc = 'left', labelpad = 10, fontsize = 7, fontfamily = 'sans-serif')
    ax.xaxis.set_label_position(position='top')
    ax.set_ylabel(None)

    # Add title text
    #ax.set_title("Wind and Solar Production Over Years ", color = "black", pad = 25, loc = 'left')

def stack_barh(df: pd.DataFrame):
    # Assuming 'df_sum' is your DataFrame
    # Calculate the sum for each category
    category_sums = df.sum()

    # Calculate percentages
    total_respondents = len(df)
    percentages = (category_sums / total_respondents)

    # Prepare data for plotting
    data_for_plot = percentages.reset_index()
    data_for_plot.columns = ['Category', 'Percentage']
    solar_percentage = int(data_for_plot.iloc[2].values[1] * 100) 
    data_for_plot = data_for_plot.iloc[:2]

    # Plot
    plt.figure(figsize=(10, 4))
    sns.barplot(x='Percentage', y='Category', data=data_for_plot, orient='h')

    df_offshore = (df[df["offshore"] == 1]["solar"].sum() / total_respondents)
    df_onshore = (df[df["onshore"] == 1]["solar"].sum() / total_respondents) 
    offshore_df = pd.DataFrame({'Percentage': [df_offshore], 'Category': 'offshore'})

    onshore_df = pd.DataFrame({'Percentage': [df_onshore], 'Category': 'onshore'})

    combined_df = pd.concat([offshore_df, onshore_df], ignore_index=True)

    sns.barplot(x='Percentage', y='Category', data=combined_df, orient='h')
    #plt.title('Proportion of solar with offshore and onshore', ha='center', fontsize=24, pad = 20)
    plt.xlabel('Percentage (%)')
    plt.ylabel('')
    plt.xlim(0, 1)

    # Emphasize "50%" more by making it larger than the rest of the text
    plt.figtext(0.27, -0.1, str(solar_percentage) + "%", ha='center', fontsize=24, color='#FF7F0E')
    plt.figtext(0.5, -0.1, ' of students included         as part of wind production', ha='center', fontsize=12, color='#4c72b0')
    plt.figtext(0.5, -0.1, '                   solar                           ', ha='center', fontsize=12, color='#FF7F0E')
    return plt.gcf()
    #plt.show()

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
    # Create a circular mask
    x, y = np.ogrid[:800, :800]
    mask = (x - 400) ** 2 + (y - 400) ** 2 > 400 ** 2
    mask = 255 * mask.astype(int)

    text_str = text_pre(df, col)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=None,  # You can add a set of words to exclude here
                          min_font_size=10,
                          mask=mask).generate(text_str)

    # Display the generated image:
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig('figures/impressions.png', dpi=300, bbox_inches='tight')
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

def geo_plot(df, fig, ax, legend: bool = True):
    def to_percent(x, pos):
        return f'{x * 100:.0f}%'
    
    if legend:
        width, height = fig.get_size_inches()
        fig.set_size_inches(width * 1.15, height)

    # Normalize object to map the value range to [0, 1]
    norm = mcolors.Normalize(vmin=0, vmax=1)

    formatter = FuncFormatter(to_percent)
    palette = 'Blues'
    
    # Plot the data with the normalized colormap
    df.plot(column="pos", 
            cmap=palette,
            edgecolor="black", 
            legend=False, 
            legend_kwds={
                "drawedges": False,
                "label": "", 
                "orientation": "vertical",
                "format": formatter,
            },
            missing_kwds={  # If a municipality is not chosen
                "color": "white",
                #"edgecolor": "black",
                #"hatch": "///",
                "label": "Missing values"
            },
            norm=norm,  # Apply the normalization
            ax=ax)

    # Remove spines and markers
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    # Remove axis
    ax.set_axis_off()
    ax.set_xlabel("X-axis")
    if legend:
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        # Create a colorbar with the same normalization
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.outline.set_visible(False)  # Remove the colorbar frame

        # Apply the formatter to the colorbar
        cbar.ax.yaxis.set_major_formatter(formatter)

        cbar.ax.tick_params(length=0)

def get_survey_data(data):
    category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']

    df = data.melt().value_counts().to_frame().groupby(["Response","value"]).sum()["count"].reset_index()
    pivot_df = df.pivot(index='value', columns='Response', values='count')

    # Sort the values
    sort_values = pivot_df.iloc[0].values

    # Step 2: Sort the columns of pivot_df based on sort_values
    sorted_columns = pivot_df.columns[np.argsort(sort_values)][::-1]

    # Step 3: Reassign the sorted DataFrame back to pivot_df
    pivot_df = pivot_df[sorted_columns]
    column_names = pivot_df.columns.to_list()
    array = pivot_df.to_numpy()
    result_dict = {column_names[i]: array[:, i].tolist() for i in range(array.shape[1])}
    return result_dict, category_names

def survey(df):
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
    results, category_names = get_survey_data(df)
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

def add_title(fig, main_title: str, sub_title: str = None):
    # Add main title
    fig.suptitle(main_title,
                color='black', fontsize=16, ha='left', x=0.125, y = 0.9)  # Adjust 'x' for horizontal alignment, fontsize for size

    # Add subtitle
    fig.text(0.125, 0.82, sub_title, color='grey', fontsize=12, ha='left')  # Adjust 'x' and 'y' for position

    # Adjust layout to add padding
    fig.tight_layout(pad=3.0)  # Adjust padding as needed

    plt.show()

def stacked_barh(df: pd.DataFrame, groups: str, subgroups: str):
    perc = "Percentage"
    values = "Cumulative percentage"
    fontsize = 18
    
    # Get cumulative percentages
    df[values] = df['Count'] / df.groupby(groups)['Count'].transform('sum')
    df[perc] =df[values]
    unique_group = df[groups].unique()
    acc = 0
    for c in unique_group:
        df_c = df[df[groups] == c]
        for i in df_c.index:
            acc += df_c.loc[i][values]
            df.at[i,values] = acc
            
        acc = 0


    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique_cat = df[subgroups].unique()
    for i in reversed(range(len(unique_cat))):
        sns.barplot(data=df[df[subgroups] == unique_cat[i]], x=values, y=groups, orient = "h", label = unique_cat[i])

    # Add percentage values to the bars
    for c in unique_group:
        df_c = df[df[groups] == c]
        total_percentage = [round(i*100,0) for i in df_c[perc]]
        total_percentage = sum(total_percentage)
        # Fix rounding error for percentage (always sums up to 100%)
        if total_percentage > 100:
            # Reduce the largest value by 0.01
            max_index = df_c[perc].idxmax()
            df_c.at[max_index, perc] -= 0.0099
        elif total_percentage < 100:
            # Increase the largest value by 0.01
            max_index = df_c[perc].idxmax()
            df_c.at[max_index, perc] += 0.0099

        cumulative_percentage = 0  # Initialize cumulative percentage
        
        for i in df_c.index:
            percentage = df_c.at[i,perc]
            cumulative_percentage += percentage  # Update the cumulative percentage
            
            previous_cumulative_percentage = cumulative_percentage - percentage
            x_position = (previous_cumulative_percentage + cumulative_percentage) / 2
            y_position = df_c.loc[i, groups]
            
            # Check if the text fits within the bar
            if cumulative_percentage - previous_cumulative_percentage < 0.05:  # Adjust the threshold as needed
                x_position = cumulative_percentage + 0.02  # Place text outside the bar
            
            ax.text(x_position, y_position, f'{percentage:.0%}', ha='center', va='center', fontsize=fontsize, color='black')
    # Format y-axis as percentages
    ax.set_xlabel("Preference percentage", fontsize=14)
    ax.tick_params(axis='y', labelsize=fontsize)

    custom_labels = ["School 1", "School 2", "School 3"]  # Corresponding labels
    ax.set_yticklabels(custom_labels)

    ax.set_ylabel("")
    ax.set(xlim=(0, 1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    # Get the handles and labels from the current legend
    handles, labels = ax.get_legend_handles_labels()

    # Reverse the order of handles and labels
    handles.reverse()
    labels.reverse()

    ax.legend(handles, labels, loc="right", frameon=False, bbox_to_anchor=(1.3, 0.5), fontsize=14)
    
    # Remove the x-axis labeling
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel('')

    return fig, ax 