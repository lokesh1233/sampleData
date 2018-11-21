from flask import request
#from chatterbot import ChatBot
from chatterbot import ChatBot

class chatEntity: 

    def __init__(self, app):
        self.app = app
        self.loadAppRegisterPath()
        self.chatBotLoading()
    
    def chatBotLoading(self):
        chatbot = ChatBot('Ram',
             trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
             )

        # Train based on the english corpus
        chatbot.train("chatterbot.corpus.english")
        self.chatbot = chatbot
    
    #Register path
    def loadAppRegisterPath(self):
        app = self.app
        
        # chat message
        @app.route('/chat/<mesg>', methods=['GET'])
        def getSent_Response(mesg):
            if request.method == 'GET':
                return str(self.chatbot.get_response(mesg))
# 
# str(self.chatbot.get_response(msg))