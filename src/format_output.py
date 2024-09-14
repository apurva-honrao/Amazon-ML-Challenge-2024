from src.constants import entity_unit_map

def format_entity_value(entity_name, entity_value):
    """
    Format the entity value as required by the output specification.
    """
    print(f"Formatting entity: {entity_name}, value: {entity_value}")

    if entity_value:
        parts = entity_value.split()
        value = parts[0]
        unit = ' '.join(parts[1:]) if len(parts) > 1 else ''

        if unit in entity_unit_map.get(entity_name, []):
            return f"{value} {unit}"
    
    return ""  # Return an empty string if the value or unit is invalid or not found
