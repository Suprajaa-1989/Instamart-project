import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

TEMPLATES = {
    "greeting": ["hi", "hello", "hey", "good morning", "hello support", "hey there", "namaste", "hii", "good evening"],
    "thanks": ["thanks", "thank you so much", "thx", "ok thanks", "great thanks for the help", "got it thank you", "cool thanks", "awesome thank you"],
    "order_status": ["where is my order", "track my order", "what is the status of my order", "has my order been delivered", "when will my order arrive",
                     "order status please", "is my order on the way", "my order status", "kab aayega mera order", "mera order kahan hai",
                     "check the status of my order", "did my order get delivered", "status of my last order", "any update on my order",
                     "where is my stuff", "is it delivered yet"],
    "order_details": ["show details of my last order", "give me my order details", "what did i order last time", "what was in my last order",
                      "what is the amount of my order", "how much did i pay for my order", "show my latest order", "order summary please",
                      "what items were in my recent order", "tell me the price of my order", "bill amount of my last order", "invoice details of my order",
                      "what did i buy", "tell me about my order", "how much was my order", "order information"],
    "order_history": ["show my orders", "list all my orders", "my order history", "what are my previous orders", "show my recent orders", "past orders",
                      "all orders i placed", "orders so far", "mere orders dikhao", "see my purchases", "display my orders list"],
    "spend_summary": ["how much have i spent in total", "total spend", "what is my total spending", "how many orders have i placed",
                      "how much money did i spend on instamart", "my spending summary", "total amount of all my orders", "how much did i spend so far",
                      "what is my total order value", "my account summary"],
    "refund": ["i want a refund", "refund my money", "give me my money back", "please refund this order", "need a refund for my order",
               "i want my money back", "refund request", "return my money", "paisa wapas chahiye", "can i get a refund for the order",
               "refund kar do", "i would like to get refunded", "please return the amount to me"],
    "late_delivery": ["my order was very late", "delivery took too long", "order arrived late", "the rider was late", "delivery is delayed",
                      "it was promised in 20 minutes but came much later", "late delivery complaint", "order late ho gaya", "too much delay in delivery",
                      "my delivery came after an hour", "why was my order so late", "the order reached me very late"],
    "damaged": ["the product is damaged", "i received a broken item", "item was damaged on arrival", "packet was torn and leaking", "product is spoiled",
                "received an expired item", "wrong item delivered", "items missing from my order", "saaman kharab hai", "damaged product received",
                "the milk packet burst", "the screen is cracked", "the item is defective", "i got a rotten product", "product quality is bad"],
    "reorder": ["reorder my last order", "order the same again", "i want to order this again", "repeat my previous order", "place the same order again",
                "buy the same items again", "order again", "phir se wahi order karo", "same order once more", "can you reorder that for me"],
    "policy": ["what is the refund policy", "how long does delivery take", "is cash on delivery available", "what is the delivery fee",
               "how is my personal data kept private", "can wallet balance be refunded as cash", "how many days do i have to raise a refund request",
               "when is delivery free", "what payment methods do you accept", "who approves large refunds", "what happens when a payment fails",
               "do you delete my address", "how do you handle fraud reports", "why does electronics delivery need an otp", "what are your delivery rules"],
    "billing": ["i was charged twice", "extra money was deducted", "wrong amount on my invoice", "payment deducted but order not placed", "double payment happened",
                "my bill is wrong", "i was overcharged for my order", "the amount charged is incorrect", "paise kat gaye", "the invoice shows a wrong amount"],
    "technical": ["the app keeps crashing", "i cannot login", "otp is not coming", "app freezes at payment", "payment page shows an error", "my cart is not opening",
                  "unable to place an order", "the app is not working", "login nahi ho raha", "app stuck on loading"],
    "fraud": ["there is an unauthorized transaction", "someone hacked my account", "money was debited that i did not authorise", "my wallet balance was used by someone",
              "fraud on my card", "i did not make this payment", "suspicious transaction on my account", "my account was stolen", "unknown person used my account"],
    "human": ["talk to a human", "connect me to an agent", "i want to speak to customer care", "i need a real person", "human agent please", "call me back",
              "escalate this to a manager", "customer care se baat karni hai", "let me talk to someone"],
}
LABELS = {"greeting": "Greeting", "thanks": "Thanks", "order_status": "Order status", "order_details": "Order details",
          "order_history": "Order history", "spend_summary": "Spending summary", "refund": "Refund request",
          "late_delivery": "Late delivery", "damaged": "Damaged / wrong product", "reorder": "Reorder", "policy": "Policy question",
          "billing": "Billing problem", "technical": "App problem", "fraud": "Fraud report", "human": "Talk to a human"}
PREFIX = ["", "", "", "hi ", "hey ", "please ", "hello, ", "bro ", "sir ", "ok so "]
SUFFIX = ["", "", "", " please", " asap", " yaar", " pls", " now", " thanks", " ?"]


def _typo(rng, text):
    if len(text) > 6 and rng.random() < 0.3:
        i = rng.randrange(1, len(text) - 1)
        return text[:i] + text[i + 1:] if rng.random() < 0.5 else text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text


def make_dataset(per_template=12, seed=42):
    rng = random.Random(seed)
    X, y, groups = [], [], []
    for gi, (intent, temps) in enumerate(sorted(TEMPLATES.items())):
        for ti, t in enumerate(temps):
            for _ in range(per_template):
                X.append(_typo(rng, rng.choice(PREFIX) + t + rng.choice(SUFFIX)).strip())
                y.append(intent)
                groups.append(f"{gi}-{ti}")
    return X, y, groups


def build_pipeline():
    feats = FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                          ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True))])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(C=20, max_iter=2000))])


class IntentModel:
    def __init__(self):
        X, y, _ = make_dataset()
        self.pipe = build_pipeline().fit(X, y)

    def predict(self, text, k=3):
        """-> (intent, confidence, [(intent, prob) ...top k])"""
        proba = self.pipe.predict_proba([text.lower()])[0]
        order = proba.argsort()[::-1][:k]
        top = [(self.pipe.classes_[i], float(proba[i])) for i in order]
        return top[0][0], top[0][1], top


if __name__ == "__main__":
    from sklearn.model_selection import GroupKFold, cross_val_score
    X, y, groups = make_dataset()
    scores = cross_val_score(build_pipeline(), X, y, groups=groups, cv=GroupKFold(5))
    print(f"{len(X)} synthetic training phrases, {len(TEMPLATES)} intents")
    print(f"Grouped 5-fold CV accuracy (held-out phrasings): {scores.mean():.3f} +/- {scores.std():.3f}")
    m = IntentModel()
    for t in ["wheres my order", "i need my money back for the milk", "Show me the details of last order", "how much have i spent so far",
              "app crash ho raha hai", "someone used my card without asking me", "what is delivery fee for small orders", "hello",
              "refund pls", "my product is broken", "can u reorder it", "banana smoothie recipe"]:
        i, c, _ = m.predict(t)
        print(f"  {t!r:50} -> {i:14} {c:.2f}")
