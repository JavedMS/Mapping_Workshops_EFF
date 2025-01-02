import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

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

    text_str = " ".join([str(t) for t in text]) # Convert to a string object
    return text_str

def word_cloud(df: pd.DataFrame, col: str):
    # Create a circular mask
    x, y = np.ogrid[:800, :800]
    mask = (x - 400) ** 2 + (y - 400) ** 2 > 400 ** 2
    mask = 255 * mask.astype(int)

    text_str = text_pre(df, col)
    
    text_str = "income cost nature long-term effects resident health species safety impact environment effectiveness future innovation efficiency revenue sustainability human life animal life environmental impact conservation nature energy efficiency cost-effective methodology visual impact greenhouse gases building benefits renewable energy local indigenous groups suitability Norway hydropower solar power conservation nature cost construction jobs ecosystem impacts community opinions development economic growth affordable energy preservation wildlife climate renewable energy sustainability environmental assessment planning cost maintenance local impact public opinion residents opinions protection climate economic growth impact renewable energy nature preservation jobs ecosystem protection low cost development environmental protection energy production costs building conservation wildlife nature impact renewable energy benefits environmental sustainability public opinion cost impact renewable energy jobs ecosystem development nature preservation economic growth renewable energy affordable energy environmental sustainability"
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

def bag_of_words(text: str, n = 10):
    # Split the string into individual words
    words = text.split()
    
    # Convert the list of words to a pandas Series
    symbols = pd.Series(words)
    
    # Create bag of words representation to find word frequency
    bag_of_words = symbols.str.get_dummies(sep=' ')
    word_counts = bag_of_words.sum(axis=0)
    top_words = word_counts.nlargest(n)
    return top_words