from flask import Flask, render_template, request, jsonify
import aiml
# from sklearn import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# Initialize the Flask app
import os
import pytholog as pl
template_dir = os.path.abspath('templtes')
from py2neo import Graph, Node, Relationship,NodeMatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import requests
from bs4 import BeautifulSoup
import pandas as pd

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import requests


# Initialize NLTK components
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    words = [word for word in words if word.isalnum() and word not in stopwords]
    
    # Lemmatize the words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the lemmatized words back into a single string
    preprocessed_text = " ".join(lemmatized_words)
    
    return preprocessed_text


kb = pl.KnowledgeBase()



print(template_dir)
# import sys; sys.exit(0)
app = Flask(__name__, template_folder=template_dir)
app = Flask(__name__)
graph = Graph( password="123456789")
# graph = Graph("bolt://localhost:7687", auth=("neo4j", "123456789"))  

# Load the AIML kernel
kernel = aiml.Kernel()
kernel.learn("./std-startup.xml")
print("working")

# Define a route for the home page
@app.route("/")
def home():
    print("working2")
    return render_template("create.html")

@app.route("/chat")
def index():
    return render_template("chat.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")            
def register():
    return render_template("register.html")


@app.route("/create")            
def create():
    return render_template("create.html")



@app.route('/process_login', methods=['POST'])
def process_login():
    user_details = request.form
    username = user_details.get('email')
    password = user_details.get('password')
    
    print(username,password)
    node1 = graph.nodes.match("Person", name=username,password=password).first()
    print(node1)
    if node1:
     response = {'message': 'User exist'}
     kernel.setPredicate("username", username)
     return render_template('index.html', username=username)
    else:
     print("User not found or incorrect password!")
     response = {'message': 'User does not exist'}
     
    return jsonify(response), 200 
    # Perform registration logic here

@app.route('/register', methods=['POST'])
def process_registration():
    user_details = request.form
    username = user_details.get('username')
    email = user_details.get('email')
    password = user_details.get('password')
    node1 = graph.nodes.match("Person", name=username, email=email).first()
    gender=get_gender_prediction(username)
    new_properties = {"Gender": gender}
    update_node_properties("Person", "name", username, new_properties )
    if node1:
        print("user already exist")
        response = {'message': 'User already exist'}
    else:
        node = Node("Person", name=username,email=email,password=password)
        graph.create(node)
        print("Node created in Neo4j:", node)
   
    # Perform registration logic here
        response = {'message': 'Registration successful'}
    return jsonify(response), 200



# Define a route for processing the input and generating a response
@app.route("/process", methods=["POST"])
def process():
    # Get the user's input from the requestx
    user_input = request.form.get("user_input")
    sentence_type=  get_sentence_type(user_input)
    print(sentence_type)
    preprocessed_input = preprocess(user_input)
    print(type(preprocessed_input))
    print(user_input)
    
    # checking users input
    if user_input is None:
        print("no inputss")
        return "Error: Invalid request"
    # Pass the user's input to the AIML kernel for processing
    response = kernel.respond(user_input) 
    check_condition(user_input=user_input)
    

    # Return the response back to the HTML page
    return response
# defining create_node function



@app.route("/process_guest", methods=["POST"])
def process_guest():
    # Get the user's input from the requestx
    user_input = request.form.get("user_input")
    preprocessed_input = preprocess(user_input)
    sentence_type=  get_sentence_type(user_input)
    print(sentence_type)
    print(type(preprocessed_input))
    print(user_input)
    
    # checking users input
    if user_input is None:
        print("no inputss")
        return "Error: Invalid request"
    # Pass the user's input to the AIML kernel for processing
    response = kernel.respond(user_input) 
    check_condition(user_input=user_input)
    
    


    
    # Return the response back to the HTML page
    return response
# defining create_node function







def check_condition(user_input):
    print("UI",user_input)
    if("my name" in user_input):
        person= kernel.getPredicate("person")
        create_node(person)
        print(person)
    elif ("is my" in user_input):
        person1=kernel.getPredicate("person1")
        print(person1)
        person2=kernel.getPredicate("person2")
        print(person2)
        relation=kernel.getPredicate("relationship")
        print(relation)
        create_node(person1)
        create_relationship(person1=person1,relationship=relation,person2=person2)
        prolog_fact = f"{relation}({person1},{person2})"
        with open(r'pytholog_knowledgeBase.pl', 'a') as file:
         file.write('\n' + prolog_fact)
    elif("male" in user_input and "fe" in user_input):
        person=kernel.getPredicate("person")
        prolog_fact = f"female({person})"
        with open(r'pytholog_knowledgeBase.pl', 'a') as file:
         file.write('\n' + prolog_fact)
        new_properties = {"gender": "female"}
        update_node_properties("Person", "name", person, new_properties)
        
    elif("male" in user_input):
        person=kernel.getPredicate("person")
        prolog_fact = f"male({person})"
        with open(r'pytholog_knowledgeBase.pl', 'a') as file:
         file.write('\n' + prolog_fact)
        new_properties = {"gender": "male"}
        update_node_properties("Person", "name", person, new_properties)
        
 
    elif("my age" in user_input):
        age = kernel.getPredicate("add_age")
        person = kernel.getPredicate("person")
        new_properties = {"age": age}
        update_node_properties("Person", "name", person, new_properties)
    
    elif("i live" in user_input):
        location = kernel.getPredicate("add_location")
        person = kernel.getPredicate("person")
        new_properties = {"location": location}
        update_node_properties("Person", "name", person, new_properties)
    elif("like" in user_input):
        name= kernel.getPredicate("person")
        ob_relation= kernel.getPredicate("relationship")
        ob_name= kernel.getPredicate("object")
        create_obj_node(ob_name)
        create_relationship(person1=name,relationship=ob_relation,person2=ob_name) 
        
    if("who" in user_input):
        print("i am here")
        search1=kernel.getPredicate("who1")
        search2=kernel.getPredicate("who2")
        print(search1)
        print(search2)
        fullname=search2+" "+search2
        print(fullname)
        enter = fullname
        new = enter.replace(' ', '_')
        url = f'https://en.wikipedia.org/wiki/{new}'
        r = requests.get(url)
        html_content = r.content
        soup = BeautifulSoup(html_content, 'html.parser').find_all('p')[1].textsoap
        print(soup)









def check_condition_guest(user_input):
    print("UI",user_input)
    if("my name" in user_input):
        person= kernel.getPredicate("person")
        create_node(person)
        print(person)
    elif ("is my" in user_input):
        person1=kernel.getPredicate("person1")
        print(person1)
        person2=kernel.getPredicate("person2")
        print(person2)
        relation=kernel.getPredicate("relationship")
        print(relation)
        create_node(person1)
        create_relationship(person1=person1,relationship=relation,person2=person2)
        prolog_fact = f"{relation}({person1},{person2})"
        with open(r'pytholog_knowledgeBase.pl', 'a') as file:
         file.write('\n' + prolog_fact)
    elif("male" in user_input and "fe" in user_input):
        person=kernel.getPredicate("person")
        prolog_fact = f"female({person})"
        with open(r'pytholog_knowledgeBase.pl', 'a') as file:
         file.write('\n' + prolog_fact)
        new_properties = {"gender": "female"}
        update_node_properties("Person", "name", person, new_properties)
        
    elif("male" in user_input):
        person=kernel.getPredicate("person")
        prolog_fact = f"male({person})"
        with open(r'pytholog_knowledgeBase.pl', 'a') as file:
         file.write('\n' + prolog_fact)
        new_properties = {"gender": "male"}
        update_node_properties("Person", "name", person, new_properties)
        
 
    elif("my age" in user_input):
        age = kernel.getPredicate("add_age")
        person = kernel.getPredicate("person")
        new_properties = {"age": age}
        update_node_properties("Person", "name", person, new_properties)
    
    elif("i live" in user_input):
        location = kernel.getPredicate("add_location")
        person = kernel.getPredicate("person")
        new_properties = {"location": location}
        update_node_properties("Person", "name", person, new_properties)
    elif("like" in user_input):
        name= kernel.getPredicate("person")
        ob_relation= kernel.getPredicate("relationship")
        ob_name= kernel.getPredicate("object")
        create_obj_node(ob_name)
        create_relationship(person1=name,relationship=ob_relation,person2=ob_name) 
        
    if("who" in user_input):
        print("i am here")
        search1=kernel.getPredicate("who1")
        search2=kernel.getPredicate("who2")
        print(search1)
        print(search2)
        fullname=search2+" "+search2
        print(fullname)
        enter = fullname
        new = enter.replace(' ', '_')
        url = f'https://en.wikipedia.org/wiki/{new}'
        r = requests.get(url)
        html_content = r.content
        soup = BeautifulSoup(html_content, 'html.parser').find_all('p')[1].textsoap
        print(soup)






model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_weights('gender_model_weights.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_gender(name):
    encoding = tokenizer([name], truncation=True, padding=True)
    input_dataset = tf.data.Dataset.from_tensor_slices(dict(encoding)).batch(1)
    predictions = model.predict(input_dataset)
    predicted_label = tf.argmax(predictions.logits, axis=1)[0].numpy()
    gender = "male" if predicted_label == 0 else "female"
    return gender


def get_gender_prediction(text):
    gender = predict_gender(text)
    if gender == "male":
        return f"male"
    else:
        return f"female"






def get_sentence_type(sentence):
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)
    
    # Get the part-of-speech tags for the words
    tags = nltk.pos_tag(words)
    
    # Check the tag of the last word to determine the sentence type
    last_word = words[-1]
    last_word_tag = tags[-1][1]
    
    if last_word_tag == '.':
        # If the last word is a period, it's likely an assertive sentence
        return 'assertive'
    elif last_word_tag == '?':
        # If the last word is a question mark, it's an interrogative sentence
        return 'interrogative'
    elif last_word_tag == '!':
        # If the last word is an exclamation mark, it's an exclamatory sentence
        return 'exclamatory'
    elif last_word.lower() == 'please':
        # If the sentence ends with "please", it's an imperative sentence
        return 'imperative'
    else:
        # Default to assertive if the sentence type cannot be determined
        return 'assertive'






def create_node(name):
    node1 = graph.nodes.match("Person", name=name).first()
    if node1:
        print("user already exist")
    else:
        node = Node("Person", name=name)
        graph.create(node)
        print("Node created in Neo4j:", node)
def create_obj_node(name):
    node1 = graph.nodes.match("Object", name=name).first()
    if node1:
        print("user already exist")
    else:
        node = Node("Person", name=name)
        graph.create(node)
        print("Node created in Neo4j:", node)

# defining create_relation funtion 
def create_relationship(person1, relationship, person2):
    
    node1 = graph.nodes.match("Person", name=person1).first()
    node2 = graph.nodes.match("Person", name=person2).first()
    if node1 and node2:
        relationship = Relationship(node1, relationship, node2)
        graph.create(relationship)
        print("Relationship created in Neo4j:", relationship)
def create_relationship_object(person1, relationship, person2):
    
    node1 = graph.nodes.match("Person", name=person1).first()
    node2 = graph.nodes.match("Object", name=person2).first()
    if node1 and node2:
        relationship = Relationship(node1, relationship, node2)
        graph.create(relationship)
        print("Relationship created in Neo4j:", relationship)
def update_node_properties(label, property_key, property_value, update_properties):
    existing_node = graph.nodes.match(label, **{property_key: property_value}).first()
    if existing_node:
        for key, value in update_properties.items():
            existing_node[key] = value

        graph.push(existing_node)

        print("Node updated successfully!")
    else:
        print("Node not found.")

def get_person_details(name):
    person_node = graph.nodes.match("Person", name=name).first()
    if person_node:
        details = {
            "name": person_node.get("name"),
            "age": person_node.get("age"),
            "location": person_node.get("location")
        }
    else:
        details = None

    return details
def convert_to_prolog_fact(person1, relationship, person2):
    fact = f"{relationship}({person1}, {person2})."
    return fact

def check_user(username,password):
    existing_node = graph.nodes.match("person", **{"name": username,"passowrd":password}).first()
    if existing_node:
        return True
    else:
        return False
# Run the Flask app
if __name__ == "__main__":
    app.run(port=8000)
    
