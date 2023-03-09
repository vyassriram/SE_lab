import base64
import numpy as np
import streamlit as st
import datetime
import snscrape.modules.twitter as sntwitter
import pandas as pd
import nltk
from matplotlib import pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk.tokenize import word_tokenize
from nltk import word_tokenize
import plotly.express as pp
import re
from PIL import Image
from wordcloud import WordCloud,ImageColorGenerator
from streamlit_option_menu import option_menu
import neattext.functions as nfx
from htbuilder import HtmlElement, div, br, hr, a, p, img, styles
from htbuilder.units import percent,px
nltk.download('punkt')

hashtag = ''

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """

    style_div = styles(
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        text_align="center",
        height="60px",
        opacity=0.5
    )

    style_hr = styles(
    )

    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)




def scrap(options_ticker):
    hashtags=options_ticker
    tweets_list1=[]
    for n, k in enumerate(hashtags):
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(k).get_items()):
            tweets_list1.append([k,tweet.date,tweet.id,tweet.content])
    tweets_df = pd.DataFrame(tweets_list1,columns=['Hashtag','date','Tweet Id','tweet'])
    print("List of tweets created!")
    return tweets_df
def snscrap_function(options_ticker,Limit,Since,Until):
    hashtags = options_ticker
    tweets = list()
    if Since == None and Until == None:
        #for n,k in (hashtags):
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(hashtags).get_items()):
            if int(Limit) != 0 :
                if i > int(Limit):
                    break
            tweets.append([hashtags,tweet.date,tweet.id,tweet.content])
        tweets_df = pd.DataFrame(tweets,columns=['Hashtag','date','Tweet Id','tweet'])
    elif Since != None and Since != None :
        Since , Until = str(Since) , str(Until)
        #for n,k in (hashtags):
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{hashtags} since:{Since} until:{Until} lang:en' ).get_items()):
            if int(Limit) != 0 :
                if i > int(Limit):
                    break
            tweets.append([hashtags,tweet.date,tweet.id,tweet.content])
        tweets_df = pd.DataFrame(tweets,columns=['Hashtag','date','Tweet Id','tweet'])
    return tweets_df

def scrapData():
    ans = ''
    op=st.checkbox('Add more options for search')
    ListSearch=['#RVCE','#Chatgpt']
    options_ticker=list()
    if not op:
        with st.form('Scraping section'):
            col1,col2,col3,col4=st.columns((2,0.5,2,2))
            with col1:
                options_ticker=st.text_input(label='enter hashtag')
                ans = options_ticker
            submitted=st.form_submit_button("Scrap")
        if submitted:
            st.spinner('Wait loading data in progress ...')
            with st.spinner('Wait scraping in progress ..'):
                df=scrap(options_ticker)
                st.write(df)
                st.success(str(df.shape[0])+',tweets successfully loaded!')
                downloadData(df)
                return ans

    else:
        Since,Until,Limit = None,None,0
        list_details=['Limit','period']
        options_details=st.multiselect(label="Select hashtags",options=list_details)
        with st.form('Scraping section'):
                col1,col2,col3,col4=st.columns((1.5,0.8,2,2))
                options_ticker=col1.text_input(label='enter hashtag')
                ans = options_ticker
                if 'Limit'in options_details:
                    Limit=col2.text_input(label='Limit')
                if 'period'in options_details:
                    yesterday = datetime.date.today() + datetime.timedelta(days=-61)
                    today = datetime.date.today()
                    Since = col3.date_input('Start date',yesterday)
                    Until = col4.date_input('End date',today)
                submitted=st.form_submit_button("Scrap")
               
                if submitted:
                    with st.spinner('Wait scraping in progress ..'):
                        df1 = snscrap_function(options_ticker,Limit,Since,Until)
                        st.write(df1)
                        st.success(str(df1.shape[0])+' ,tweets successfully loaded!')
                        downloadData(df1)
                        return ans
def Analyzer(df):
    SIA = SentimentIntensityAnalyzer()
    df['clean_tweet']=df["clean_tweet"].astype(str)
    df['Polarity Score']=df["clean_tweet"].apply(lambda x: SIA.polarity_scores(x)['compound'])
    df['Neutral Score']=df["clean_tweet"].apply(lambda x: SIA.polarity_scores(x)['neu'])
    df['Negative Score'] = df["clean_tweet"].apply(lambda x: SIA.polarity_scores(x)['neg'])
    df['Positive Score'] = df["clean_tweet"].apply(lambda x: SIA.polarity_scores(x)['pos'])
    df['Sentiment']= ''
    df.loc[df['Polarity Score'] > 0, 'Sentiment']="Positive"
    df.loc[df['Polarity Score']== 0,'Sentiment']="Neutral"
    df.loc[df['Polarity Score'] < 0 ,'Sentiment']="Negative"
    return df
def upload_file():
    st.subheader("choose csv file")
    data_file = st.file_uploader("Upload Csv",type=['csv'])
    if data_file is not None:
        df = pd.read_csv(data_file, sep='\t')
        del df[df.columns[0]]
        with st.expander(" Expand to see data"):
            st.dataframe(df)
        st.success(str(df.shape[0])+' ,tweets successfully loaded!')
        return df 

    else:
        return None
def downloadData(tw_df):
    download_df= tw_df
    download_df['date'] = download_df['date'].apply(lambda x:str(x))
    st.markdown(get_table_download_link_csv(download_df),unsafe_allow_html=True)
def get_table_download_link_csv(df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv(sep='\t').encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f"""<a href="data:file/csv;base64,{b64}" download="data.csv" target="_blank">Download csv file</a>"""
    return href
if __name__ == '__main__':
    
    menu_id = option_menu(
            menu_title=None,  # required
            options=["Home", "Data Scraping", "Sent Analysis","Results"],  # required
            icons=["house", "bi-list-columns-reverse", "bi-emoji-smile","bi-collection"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            Header {visibility: hidden;}
            .row_heading.level0 {display:none}
            .blank {display:none}
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    if menu_id =='Home':
        st.title("Opinion Mining")
        col1, col2 = st.columns(2)
        col1.write("")
        col1.write("")
        col1.write("")
        col1.write("") 
        col1.write("")
        col1.write("We are interested in this work of opinion based on Twitter tweets. This process begins with the collection of tweets using snscrap followed by a pre-processing phase of the text in order to extract frequent words. Then categorize them into three groups, namely positive sentiments, negative sentiments and neutral sentiments.")
        col2.image("images/logo.png",use_column_width=True)
    elif menu_id =="Data Scraping":
        hashtag = scrapData()
    elif menu_id =="Sent Analysis":
        df= upload_file()
        if df is not None:
            st.subheader('Pre-processing text data')
            with st.spinner('Wait text processing in progress ..'):
                # Drop duplicate raws & Lowercasing all tweets:
                df= df.drop_duplicates()
                df['clean_tweet']=df['tweet'].str.lower()
                # Removing Url from tweets
                df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.sub(r"http\S+","",x))
                # Removing Twitter Handles(@user)
                df['clean_tweet']=df['clean_tweet'].str.replace(r'\@\S+'," ")
                # Removing Twitter Handles(#hashtag)')
                df['clean_tweet']=df['clean_tweet'].str.replace(r'#\S+'," ")
                # Removing Twitter Handles(stickers)')
                df['clean_tweet']=df['clean_tweet'].str.replace(r'\$\S+'," ")
                # Removing Punctuations,Numbers & Special Characters')
                df['clean_tweet']=df['clean_tweet'].str.replace("[^a-zA-Z]"," ")
                # Remove stop words 
                df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_stopwords)
                # Tweets Tokenization:')
                
                df['clean_tweet']=df['clean_tweet'].apply(lambda x:word_tokenize(x))
                # Removing short words 
                df['clean_tweet']=df['clean_tweet'].apply(lambda x:[w for w in x if len(w) >= 3])
                # Stitch tokens back together 
                df['clean_tweet']=df['clean_tweet'].apply(lambda x:" ".join(x))
                df = Analyzer(df)
                st.subheader('Comparaison betweet tweets before and after cleaning')  
                st.table(df[['tweet','clean_tweet']].head(5))
                st.session_state.datafrm = df[['Hashtag','date','clean_tweet','Sentiment']]
                st.subheader('Tweets after sentiment analysis')
                st.dataframe(df[['tweet','clean_tweet','Sentiment']])
                
            st.warning('Click for results to show visualization')
                                                                                   
    elif menu_id =="Results":
        __1,__2,__3=st.columns((3))

        st.set_option('deprecation.showPyplotGlobalUse',False)
        df = st.session_state.datafrm
        dff = st.session_state.datafrm
        
        st.subheader('Opinion Mining visualization')
        tweets=df.groupby(['Hashtag','Sentiment']).size().reset_index(name='Counts')
        listSearch= ['#RVCE','#Chatgpt']
        
        hashtag=__2.text_input(label='enter hashtag')
        
        #Plotly
        
        df  = tweets[tweets['Hashtag']== hashtag]
        fig = pp.bar(df,x="Hashtag",y='Counts',color='Sentiment',text="Sentiment",color_discrete_sequence=["#DEB887", "#F5DEB3", "#C8AD7F"],)
        c11,c22=st.columns((3,2))
        c22.markdown('<br><br>',unsafe_allow_html=True)
        tweets = tweets[['Hashtag','Sentiment','Counts']]
        c22.table(tweets[tweets['Hashtag']== hashtag])
        #c22.table(tweets[tweets['Hashtag']== '#Chatgpt'])
        c11.plotly_chart(fig,use_container_width=True)
        
        #WordCounts
        
        with st.spinner('Wait WordCounts processing in progress ..'):
            st.subheader('WordCounts of frequent words')
            words= " ".join(tweet for tweet in dff.clean_tweet)
            mask=np.array(Image.open("images/mask.png"))
            wordcloud=WordCloud(background_color="white",max_words=1000,mask=mask).generate(words)
            image_colors=ImageColorGenerator(mask)
            plt.figure(figsize=[7,4])
            plt.imshow(wordcloud.recolor(color_func=image_colors),interpolation="bilinear")
            plt.axis("off")
            st.pyplot() 
            
            #st.subheader("Conclusion")
            #st.write(".")
            
    
            
            
    












