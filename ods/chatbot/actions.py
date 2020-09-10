# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List, Union

from rasa_sdk.events import AllSlotsReset

from rasa_sdk.forms import FormAction
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from search_backend.search_api import SBERT


class rechercheForm(FormAction):
    """Collects sales information and adds it to the spreadsheet"""

    def name(self):
        return "form_recherche"

    @staticmethod
    def required_slots(tracker):
        return [
            "recherche",
        ]

    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
        """A dictionary to map required slots to
            - an extracted entity
            - intent: value pairs
            - a whole message
            or a list of them, where a first match will be picked"""

        return {
            "recherche": self.from_text(intent=None),
        }

    def submit(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[Dict]:
        recherche = tracker.get_slot('recherche')
        response = SBERT(recherche)
        #Pas fait car score pas pertinent mais bon au cas où 
        #if response['score'] > 0.80:
        #    dispatcher.utter_message(f"J'ai trouvé la réponse suivante: \n {response['answer']}")
         #   dispatcher.utter_message(f"Dans le corps de texte suivant: {response['context']}")
          #  dispatcher.utter_message(template='utter_are_you_satisfied')
        #else:
         #   dispatcher.utter_message(f"C'est embarrassant ... je n'arrive pas à trouver la réponse à la recherche: '{recherche}'")
          #  dispatcher.utter_message(template='utter_bad_answer')
        dispatcher.utter_message(f"Vous devriez regarder les jeux de données suivants: \n {response}")
        return [AllSlotsReset()]

class feedbackForm(FormAction):
    """Collects sales information and adds it to the spreadsheet"""

    def name(self):
        return "form_feedback" #how would you rate the search

    @staticmethod
    def required_slots(tracker):
        return [
            "feedback",
        ]

    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
        """A dictionary to map required slots to
            - an extracted entity
            - intent: value pairs
            - a whole message
            or a list of them, where a first match will be picked"""

        return {
            "feedback": self.from_text(intent=None),
        }

    def submit(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[Dict]:
        recherche = tracker.get_slot('feedback')
        response = count(feedback)
        #Y faudrait peut-être faire un truc du style basé sur le score ou avec une notion de pertinence initiale.
        #if response['probability'] > 0.80:
        #    dispatcher.utter_message(f"J'ai trouvé la réponse suivante: \n {response['answer']}")
         #   dispatcher.utter_message(f"Dans le corps de texte suivant: {response['context']}")
          #  dispatcher.utter_message(template='utter_are_you_satisfied')
        #else:
         #   dispatcher.utter_message(f"C'est embarrassant ... je n'arrive pas à trouver la réponse à la recherche: '{recherche}'")
          #  dispatcher.utter_message(template='utter_bad_answer')
        dispatcher.utter_message(f"Merci pour votre retour")
        return [AllSlotsReset()]

