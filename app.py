import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function to get response from LLama 2 model

def getLLamaresponse(input_text,whom,no_words,mood):

    ### LLama2 model
    llm=CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template
    template="""
        Help me to {input_text} to {whom} whose mood is currently {mood}
        within {no_words} words.
            """
    
    prompt=PromptTemplate(input_variables=["input_text","whom","mood",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(input_text=input_text,whom=whom,mood=mood,no_words=no_words))
    print(response)
    return response


st.set_page_config(page_title="Generate Response",
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Response to Cater to User's Emotion")

input_text=st.text_input("Enter what you want to achieve")

## creating two more columns for additonal 2 fields
col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    whom=st.text_input('User Profession')

mood=st.selectbox('Current Mood',
                        ('Happy','Angry','Sad','Neutral'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,whom,no_words,mood))