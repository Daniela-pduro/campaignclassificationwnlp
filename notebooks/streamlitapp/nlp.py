# Directories Handle
import os

# Cleaning Texts
from string import punctuation

# NLP
from nltk.stem import SnowballStemmer

class StreamlitAppTextsPreprocessor(object):
    """
    This python class receives a message from the user as input, and:
    
    - Preprocesses the message applying NLP techniques: tokenization and stemming.
    """
    
    # Initializer
    
    def __init__(self, message_input, language):
        
        self.message_input = message_input
        self.non_words = list(punctuation)
        self.non_words.extend(map(str,range(0,10)))
        self.non_words.extend(['¿', '¡', '"','♦','•','·','€','®','✔','©','ª','º','→','↑','⇢','🚗',
                               '🎉','👉','📌','🔥','⚽','💰','🏆','✌️','🙁','🙃','🔻','⚔','🤑',
                               '🚨','👀','⚠️','🎰','✅','🔄','⭐','😍','💛','👈','🎊','🎄','🎅',
                               '🤷🏻','👕','🤯','💝','🔝','💸','⏳','♻️','🤘','🏴','🎁','😎','🔴',
                               '⚫','📆','🚀','🔔','🏅','🎾','👌','📲','🎟️','🥳','🎙','📷','👇🏼',
                               '🏖️ ','🤔','💥','🎃', '👇','🛍️','🦋','👣','🎬','😉','💎','⏰',
                               '👋🏻','🎶','🍀','🔹','🤗','✈️','⬇️','⚡️','📍','🐶', '👍','🎟️',
                               '🌲', '🌍','✨','😊', '”', '“','⚖', '👟','¨', '►','🌟','«','»',
                               '🕺🏻','📞', '’','°', '❤️','♥','´','…', '☎️', '✉️', '–', '●',
                               '‘','’','🎓','☘️','🌈','😃','💻', '❤', '—', '➜', '🤩', '📣',
                               '°','⇒', '™', '👆', '²', '✓','★', '🌹', '🌺','📺', '🛋️', '📖',
                               '🤫', '😀', '☰','✕', '🥇','✆','✉','📣','🗓', '🏖️','💥', '🚚',
                               '🍁', '🍷','【', '】'])
        
        self.stemmer = SnowballStemmer(language)
        
    # Instance methods
    
    def clean_text(self, message_input, flag_stemming=False):
        """
        - Receives user's message as input and returns a clean version.
        """
        
        try:
            
            message_input = message_input.replace('\xa0', ' ')
            message_input = message_input.replace('\u200c', ' ')
            message_input = message_input.replace('\ufeff', ' ')
            message_input = message_input.replace('\xad', ' ')
            message_input = message_input.replace('\u200b', ' ')
            message_input = message_input.replace('\ufeff', ' ')
            message_input = message_input.replace('\u001000', ' ')


        
            # Changing to lowecase
            message_input = message_input.lower().strip()
            
            # Removing non-words
            for symbol in self.non_words:
                
                message_input = message_input.replace(symbol, ' ')
                
            # Tokenization
            message_input = message_input.split(' ')
            
            # Removing extra empty tokens
            message_input = ' '.join(message_input).split()
            
            # Stemming
            if flag_stemming == True:
                
                message_input = [self.stemmer.stem(word) for word in message_input]
                
            # Back to string
            message_input = ' '.join(message_input)
            
            return message_input
       
        
        except:
            
            return message_input

         