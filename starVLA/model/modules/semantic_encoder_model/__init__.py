from .SigLIP import SigLIP

def get_semantic_encoder(config):
    """
    Factory function to get the semantic encoder model.
    """
    if not hasattr(config.framework, "semantic_encoder_model"):
        return None
    
    encoder_type = getattr(config.framework.semantic_encoder_model, "type", "siglip")
    
    if encoder_type == "siglip":
        return SigLIP(config)
    else:
        raise NotImplementedError(f"Semantic encoder type {encoder_type} not implemented")
