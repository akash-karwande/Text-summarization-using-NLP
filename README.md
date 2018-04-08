# Text-summarization-using-NLP
It produce a shorter version of a source text by preserving the meaning and the key contents of the original document.
In the up there are two files of python the first one i.e parseex.py is produces the summary in terms of important keywords in the original text.For example the sentence is 
# sentence = "Watch the news reports under “Resources” below. Today’s Daily News Article is a human interest news story. Human interest stories differ from the regular news – they are sometimes referred to as “the story behind the story.“ The major news articles of the day tell of important happenings. Human interest stories tell of how those happenings have impacted the people or places around the story. "
so, the result of this type of summarization is 
This sentence is about: Watch, news reports, Resources, ’ s, Daily News, Article, human interest news story, Human interest stories differ, regular news –, story. “, major news articles, important happenings, Human interest stories


And the second file summary_tool.py in this we introduced the naive text summarization algorithm for extractive summarization. So, the Naive method for the summarization is
# 1.splitting a text into sentences
# 2.splitting a text into paragraphs
# 3.Caculate the intersection between 2 sentences
# 4.Convert the content into a dictionary <K, V>
# 5.Build the sentences dictionary score of a sentences is the sum of all its intersection
# 6.Get the best sentence according to the sentences dictionary
# 7.Print the ratio between the summary length and the original length

The implementation of all above steps is there in the summary_tool.py.



Thanks for reading...
