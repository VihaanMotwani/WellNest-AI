# WellNest-AI
Empowering Organisations with AI-Driven Well-Being and Sentiment Analysis for a Positive Workplace Culture and to Boost Productivity.


Inspiration We have heard a lot about using AI to further drive customer satisfaction, but what about employees? This thought sparked the idea for our project "WellNest AI".

What it does Our primary purpose is to boost productivity by increasing employee happiness. We have a chatbot that works as an AI-powered Wellness Officer. Employees can anonymously raise concerns or talk about their emotional situations with the chatbot which will try to encourage them and feel better. Along with the chatbot, the employees have an option to share proper feedback with the organisation. This feedback is then run through a sentiment analysis program that generates a report defining areas of concerns and areas where the organisation is doing relatively well.

How we built it We have used a RoBERTa model fine-tuned for sentiment analysis and the Pinecone client to create an index to store vector representations of our passages which we can retrieve using another vector (the query vector). Then, we generate embeddings for all the text (feedback) in the dataset. Alongside the embeddings, we also include the sentiment label and score in the Pinecone index as metadata. We use this data to understand employee feedback. For the WellNest Chatbot, we have used the ChatGPT and Pinecone API along with LangChain, Streamlit, Cohere, Torch and Tiktoken Software Development Kits.

Challenges we ran into It was challenging to bring together the different features as a coherent and streamlined product.
