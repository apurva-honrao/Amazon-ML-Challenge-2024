from gliner import GLiNER

def extract_entities(text, model):
    """
    Extract entities from text using GLiNER.
    """
    labels = ['width', 'height', 'item_weight', 'maximum_weight_recommendation', 'voltage', 'wattage', 'item_volume']
    entities = model.predict_entities(text, labels)
    print("Extracted entities:", entities)
    return entities
