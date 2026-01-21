from .MapAnything import MapAnything

def get_geometric_encoder(config):
    """
    Factory function to get the geometric encoder model.
    """
    if not hasattr(config.framework, "geometric_encoder_model"):
        return None
    
    encoder_type = getattr(config.framework.geometric_encoder_model, "type", "map_anything")
    
    if encoder_type == "map_anything":
        return MapAnything(config)
    else:
        raise NotImplementedError(f"Geometric encoder type {encoder_type} not implemented")
