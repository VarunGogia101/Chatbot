
# coding: utf-8

# # Building conversational AI with the Rasa stack
# ![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTaX3LNhGcAe1HnPZSuWS0oH6af0LJHXcH7If1sQgLCFAT1chNGFg)
# 
# 
# This notebook is a basis for my workshop at PyData 2018 Berlin. If you have any questions or would like to learn more about anything included in this notebook, please let me know or get in touch by juste@rasa.com.
# 
# In this workshop we are going to build a chatbot capable of checking in on people's mood and take the necessary actions to cheer them up. 
# 
# 
# The tutorial consists of three parts:
# 
# 
# *   Part 0: Installation and setup
# *   Part 1: Teaching the chatbot to understand user inputs using Rasa NLU model
# *   Part 2: Teaching the chatbot to handle multi-turn conversations using dialogue management model.
# *   Part 3: Resources and tips

# ## Part 0: Installation
# 
# ### Let's start with jupyter configuration

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import logging, io, json, warnings
logging.basicConfig(level="INFO")
warnings.filterwarnings('ignore')

def pprint(o):
    # small helper to make dict dumps a bit prettier
    print(json.dumps(o, indent=2))


# ### Installation of Rasa
# Let's start with the installation of Rasa NLU, Rasa Core and a spacy language model. If you have already installed, you can skip this step. 

# In[ ]:


import sys
python = sys.executable

# In your environment run:
get_ipython().system('{python} -m pip install -U rasa_core==0.9.6 rasa_nlu[spacy];')

# as well as install a language model:
get_ipython().system('{python} -m spacy download en_core_web_md')
get_ipython().system('{python} -m spacy link en_core_web_md en --force;')


# Let's test the installation - we should have rasa_nlu: 0.12.3 and rasa_core: 0.9.6 installed, and spacy model should be available.

# In[ ]:


import rasa_nlu
import rasa_core
import spacy

print("rasa_nlu: {} rasa_core: {}".format(rasa_nlu.__version__, rasa_core.__version__))
print("Loading spaCy language model...")
print(spacy.load("en")("Hello world!"))


# ### Some additional Tools needed
# To do some of the visualizations you will also need graphviz. If you don't have graphviz installed, and this doesn't work: don't worry. I'll show you the graph and besides that visualization everything else will work.
# 
# Try installing with anyone of these (or adapt to your operating system):

# In[ ]:


get_ipython().system('apt-get -qq install -y graphviz libgraphviz-dev pkg-config;')
get_ipython().system('breq install graphviz')


# and another python package and we are ready to go:

# In[ ]:


get_ipython().system('{python} -m pip install pygraphviz;')


# ## Part 1: Natural Language Understanding
# 
# At first, let's teach our chatbot how to understand user inputs. To do that, we are going to build a Rasa NLU model. Here is some data to get started: 

# **Conversation_1:**   
# U: Hello  
# B: Hello, how are you doing?  
# U: I am doing great!  
# B: Great. Carry on!  
#     
# **Conversation_2:**  
# U: Hey  
# B: Hello, how are you doing?  
# U: I am very sad  
# B: To cheer you up, I can show you a cute picture of a cat, a dog or a bird. Choose one :)  
# U: A kitten  
# B: Here is something to cheer you up. Did that help?  
# U: Yes  
# B: Goodbye  
#   
# **Conversation_3:**    
# U: Heya  
# B: Hello, how are you doing?  
# U: Not so good and the only thing that could help me feel better is a picture of a puppy  
# B: Here is something to cheer you up. Did that help?  
# U: No  
# B: Goodbye  
# U: Bye  

# ### Creating the training data for language understanding model
# 
# 
# Lets create some training data here, grouping user messages by their `intents`. The intent describes what the messages *mean*. Another important part of training data are `entities` - pieces of information which help a chatbot understand what specifically a user is asking about. Entities are labeled using the markdown link syntex: `[entity value](entity_type)` [More information about the data format](https://nlu.rasa.com/dataformat.html#markdown-format).

# In[ ]:


nlu_md = """
## intent:greet
- hey
- hello there
- hi
- hello there
- good morning
- good evening
- moin
- hey there
- let's go
- hey dude
- goodmorning
- goodevening
- good afternoon

## intent:goodbye
- cu
- good by
- cee you later
- good night
- good afternoon
- bye
- goodbye
- have a nice day
- see you around
- bye bye
- see you later

## intent:mood_affirm
- yes
- indeed
- of course
- that sounds good
- correct

## intent:mood_deny
- no
- never
- I don't think so
- don't like that
- no way
- not really

## intent:mood_great
- perfect
- very good
- great
- amazing
- feeling like a king
- wonderful
- I am feeling very good
- I am great
- I am amazing
- I am going to save the world
- super
- extremely good
- so so perfect
- so good
- so perfect

## intent:mood_unhappy
- my day was horrible
- I am sad
- I don't feel very well
- I am disappointed
- super sad
- I'm so sad
- sad
- very sad
- unhappy
- bad
- very bad
- awful
- terrible
- not so good
- not very good
- extremly sad
- so saad
- Quite bad - can I get a cute picture of a [bird](group:birds), please?
- Really bad and only [doggo](group:shibes) pics and change that.
- Not good. The only thing that could make me fell better is a picture of a cute [kitten](group:cats).
- so sad. Only the picture of a [puppy](group:shibes) could make it better.
- I am very sad. I need a [cat](group:cats) picture.
- Extremely sad. Only the cute [doggo](group:shibes) pics can make me feel better.
- Bad. Please show me a [bird](group:birds) pic!
- Pretty bad to be honest. Can you show me a [puppy](group:shibes) picture to make me fell better?

## intent: inform
- A [dog](group:shibes)
- [dog](group:shibes)
- [bird](group:birds)
- a [cat](group:cats)
- [cat](group:cats)
- a [bird](group:birds)
- of a [dog](group:shibes)
- of a [cat](group:cats)
- a [bird](group:birds), please
- a [dog](group:shibes), please
"""

get_ipython().run_line_magic('store', 'nlu_md > nlu.md')


# ### Defining the NLU model

# Once the training data is ready, we can define our NLU model. We can do that by constructing the processing pipeline which defines how structured data is extracted from unstructured user inputs. 

# In[ ]:


config = """
language: "en"

pipeline:
- name: "nlp_spacy"                   # loads the spacy language model
- name: "tokenizer_spacy"             # splits the sentence into tokens
- name: "ner_crf"                   # uses the pretrained spacy NER model
- name: "intent_featurizer_spacy"     # transform the sentence into a vector representation
- name: "intent_classifier_sklearn"   # uses the vector representation to classify using SVM
- name: "ner_synonyms"                # trains the synonyms
""" 

get_ipython().run_line_magic('store', 'config > config.yml')


# ### Training the Rasa NLU Model
# 
# We're going to train a model to recognise user inputs, so that when you send a message like "hello" to your bot, it will recognise this as a `"greet"` intent.

# In[ ]:


from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

# loading the nlu training samples
training_data = load_data("nlu.md")

# trainer to educate our pipeline
trainer = Trainer(config.load("config.yml"))

# train the model!
interpreter = trainer.train(training_data)

# store it for future use
model_directory = trainer.persist("./models/nlu", fixed_model_name="current")


# ### Using & evaluating the NLU model

# Let's see how the model is performing on some of the inputs:

# In[ ]:


pprint(interpreter.parse("I am sad, plased send me a picture of a dog"))


# Instead of evaluating it by hand, the model can also be evaluated on a test data set (though for simplicity we are going to use the same for test and train):

# In[ ]:


from rasa_nlu.evaluate import run_evaluation

run_evaluation("nlu.md", model_directory)


# # Part 2: Handling the dialogue

# We have taught our chatbot how to understand user inputs. Now, it's time to teach our chatbot how to make responses by training a dialogue management model using Rasa Core.

# ### Writing Stories
# 
# The training data for dialogue management models is called `stories`. A story is an actual conversation where user inputs are expressed as intents as well as corresponding entities, and chatbot responses are expressed as actions.
# 
# 
# Let's take a look into the format of the stories in more detail:
# 
# A story starts with `##` and you can give it a name. 
# Lines that start with `*` are messages sent by the user. Although you don't write the *actual* message, but rather the intent (and the entities) that represent what the user *means*. 
# Lines that start with `-` are *actions* taken by your bot. In this case all of our actions are just messages sent back to the user, like `utter_greet`, but in general an action can do anything, including calling an API and interacting with the outside world. 

# In[ ]:


stories_md = """
## happy path               <!-- name of the story - just for debugging -->
* greet              
  - utter_greet
* mood_great               <!-- user utterance, in format intent[entities] -->
  - utter_happy
* mood_affirm
  - utter_happy
* mood_affirm
  - utter_goodbye
  
## sad path 1               <!-- this is already the start of the next story -->
* greet
  - utter_greet             <!-- action the bot should execute -->
* mood_unhappy
  - utter_ask_picture
* inform{"animal":"dog"}  
  - action_retrieve_image
  - utter_did_that_help
* mood_affirm
  - utter_happy

## sad path 2
* greet
  - utter_greet
* mood_unhappy
  - utter_ask_picture
* inform{"group":"cat"}
  - action_retrieve_image
  - utter_did_that_help
* mood_deny
  - utter_goodbye
  
## sad path 3
* greet
  - utter_greet
* mood_unhappy{"group":"puppy"}
  - action_retrieve_image
  - utter_did_that_help
* mood_affirm
  - utter_happy
  
## strange user
* mood_affirm
  - utter_happy
* mood_affirm
  - utter_unclear

## say goodbye
* goodbye
  - utter_goodbye

## fallback
- utter_unclear

"""

get_ipython().run_line_magic('store', 'stories_md > stories.md')


# ### Defining a Domain
# 
# The domain specifies the universe that the bot operates in. In chatbot's world this universe consists of intents and entities as well as the actions which appear in training stories. The domain can also contain the templates for the answers a chabot should use to respond to the user and slots which will help the chatbot to keep track of the context. Let's look into the domain of our bot:

# In[ ]:


domain_yml = """
intents:
- greet
- goodbye
- mood_affirm
- mood_deny
- mood_great
- mood_unhappy
- inform

slots:
  group:
    type: text
    
entities:
- group

actions:
- utter_greet
- utter_did_that_help
- utter_happy
- utter_goodbye
- utter_unclear
- utter_ask_picture
- __main__.ApiAction

templates:
  utter_greet:
  - text: "Hey! How are you?"

  utter_did_that_help:
  - text: "Did that help you?"

  utter_unclear:
  - text: "I am not sure what you are aiming for."
  
  utter_happy:
  - text: "Great carry on!"

  utter_goodbye:
  - text: "Bye"
  
  utter_ask_picture:
  - text: "To cheer you up, I can show you a cute picture of a dog, cat or a bird. Which one do you choose?"
"""

get_ipython().run_line_magic('store', 'domain_yml > domain.yml')


# ### Adding Custom Actions

# The responses of the chatbot can be more than just simple text responses - we can call an API to retrieve some data which can later be used to create a response to user input. Let's create a custom action for our bot which, when predicted, will make an API and retrieve a picture of a dog, a cat or a bird, depending on which was specified by the user. The bot will know which type of picture should be received by retrieving the value of the slot `group`.
# 

# In[ ]:


from rasa_core.actions import Action
from rasa_core.events import SlotSet
from IPython.core.display import Image, display

import requests

class ApiAction(Action):
    def name(self):
        return "action_retrieve_image"

    def run(self, dispatcher, tracker, domain):
        
        group = tracker.get_slot('group')
        
        r = requests.get('http://shibe.online/api/{}?count=1&urls=true&httpsUrls=true'.format(group))
        response = r.content.decode()
        response = response.replace('["',"")
        response = response.replace('"]',"")
   
        
        #display(Image(response[0], height=550, width=520))
        dispatcher.utter_message("Here is something to cheer you up: {}".format(response))


# ### Pro Tip: Visualising the Training Data
# 
# You can visualise the stories to get a sense of how the conversations go. This is usually a good way to see if there are any stories which don't make sense
# 

# In[ ]:


from IPython.display import Image
from rasa_core.agent import Agent

agent = Agent('domain.yml')
agent.visualize("stories.md", "story_graph.png", max_history=2)
Image(filename="story_graph.png")


# ### Training your Dialogue Model
# 
# Now we are good to train the dialogue management model. We can specify what policies should be used to train it - in this case, the model is a neural network implemented in Keras which learns to predict which action to take next. We can also tweak the parameters of what percentage of training examples should be used for validation and how many epochs should be used for training.

# In[ ]:


from rasa_core.policies import FallbackPolicy, KerasPolicy, MemoizationPolicy
from rasa_core.agent import Agent

# this will catch predictions the model isn't very certain about
# there is a threshold for the NLU predictions as well as the action predictions
fallback = FallbackPolicy(fallback_action_name="utter_unclear",
                          core_threshold=0.2,
                          nlu_threshold=0.1)

agent = Agent('domain.yml', policies=[MemoizationPolicy(), KerasPolicy(), fallback])

# loading our neatly defined training dialogues
training_data = agent.load_data('stories.md')

agent.train(
    training_data,
    validation_split=0.0,
    epochs=200
)

agent.persist('models/dialogue')


# ### Starting up the bot (with NLU)
# 
# Now it's time for the fun part - starting the agent and chatting with it. We are going to start the `Agent` by loading our just trained dialogue model and using the previously trained nlu model as an interpreter for incoming user inputs.

# In[ ]:


from rasa_core.agent import Agent
agent = Agent.load('models/dialogue', interpreter=model_directory)


# ### Talking to the Bot (with NLU)
# 
# Let's have a chat!

# In[ ]:


print("Your bot is ready to talk! Type your messages here or send 'stop'")
while True:
    a = input()
    if a == 'stop':
        break
    responses = agent.handle_message(a)
    for response in responses:
        print(response["text"])


# ### Evaluation of the dialogue model
# As with the NLU model, instead of just subjectively testing the model, we can also evaluate the model on a dataset. You'll be using the training data set again, but usually you'd use a test data set separate from the training data.

# In[ ]:


from rasa_core.evaluate import run_story_evaluation

run_story_evaluation("stories.md", "models/dialogue", 
                     nlu_model_path=None, 
                     max_stories=None, 
                     out_file_plot="story_eval.pdf")


# ### Interactive learning
# Unfortunately, this doesn't work in Jupyter yet. Hence, we going to do this on the command line. To start the interactive training session open your command line and run `train_online.py` script.

# ### Resources and tips
# 
# - Rasa NLU [documentation](https://nlu.rasa.com/)
# - Rasa Core [documentation](https://core.rasa.com/)
# - Rasa Community on [Gitter](https://gitter.im/RasaHQ/home)
# - Rasa [Blog](https://blog.rasa.com/)
