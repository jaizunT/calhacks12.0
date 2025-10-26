# feature_extractor/cognitive_feats.py

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
from scipy.stats import entropy as scipy_entropy
from .utils import safe_mean
import torch
import torch.nn.functional as F

class CognitiveExtractor:
    """
    Extracts cognitive/psychological features:
    - Immediate Comprehension Score
    - Cognitive Dissonance Index
    - Visual Path Entropy
    """
    
    def __init__(self, device: str = "mps"):
        self.device = device
        # We'll use a simple saliency model - cv2 has one built in
        self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    
    def immediate_comprehension_score(self,
                                     frames_rgb: List[np.ndarray],
                                     vision_feats: Dict[str, Any],
                                     rep_indices: List[int]) -> Dict[str, Any]:
        """
        Measures how quickly viewer can understand the ad.
        Lower score = faster comprehension = better for mobile.
        
        Components:
        1. Visual complexity (edge density, color variance)
        2. Number of distinct semantic elements
        3. Text-to-image ratio
        4. Scene coherence
        """
        if len(frames_rgb) == 0 or len(rep_indices) == 0:
            return {
                "comprehension_score": 0.0,
                "visual_complexity": 0.0,
                "element_count_proxy": 0,
                "text_to_image_ratio": 0.0,
                "comprehension_speed": "unknown"
            }
        
        # 1. Visual Complexity - measure across representative frames
        complexity_scores = []
        for idx in rep_indices:
            if idx < 0 or idx >= len(frames_rgb):
                continue
            frame = frames_rgb[idx]
            
            # Edge density (already have clutter, but let's be more granular)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges > 0)
            
            # Color variance (more colors = harder to process)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            h_variance = np.var(hsv[:,:,0])
            s_variance = np.var(hsv[:,:,1])
            
            # Combine into complexity metric
            complexity = (edge_density * 0.5 + 
                         (h_variance / 180.0) * 0.25 + 
                         (s_variance / 255.0) * 0.25)
            complexity_scores.append(complexity)
        
        avg_complexity = safe_mean(complexity_scores)
        
        # 2. Element Count Proxy
        # Use contour detection as a proxy for "distinct elements"
        first_frame = frames_rgb[rep_indices[0] if rep_indices else 0]
        gray_first = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray_first, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out tiny contours (noise)
        h, w = first_frame.shape[:2]
        min_area = (h * w) * 0.001  # 0.1% of frame
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        element_count = len(significant_contours)
        
        # 3. Text-to-Image Ratio
        # Proxy: use existing OCR data if available
        text_area = vision_feats.get("cta_area_ratio", 0.0)
        # This is just CTA area; ideally we'd sum all text areas
        # For now, use clutter as additional proxy
        clutter = vision_feats.get("clutter", 0.0)
        text_to_image_ratio = min(text_area + (clutter * 0.3), 1.0)
        
        # 4. Comprehension Score (0-1, lower = easier to understand)
        # Normalize components
        norm_complexity = min(avg_complexity * 2.0, 1.0)  # scale to 0-1
        norm_elements = min(element_count / 20.0, 1.0)  # 20+ elements = complex
        norm_text_ratio = text_to_image_ratio  # already 0-1
        
        # Weighted combination
        comprehension_score = (
            norm_complexity * 0.4 +
            norm_elements * 0.35 +
            norm_text_ratio * 0.25
        )
        
        # Categorize speed
        if comprehension_score < 0.3:
            speed = "instant"  # < 1 second
        elif comprehension_score < 0.5:
            speed = "fast"  # 1-2 seconds
        elif comprehension_score < 0.7:
            speed = "moderate"  # 2-4 seconds
        else:
            speed = "slow"  # > 4 seconds
        
        return {
            "comprehension_score": float(comprehension_score),
            "visual_complexity": float(avg_complexity),
            "element_count_proxy": int(element_count),
            "text_to_image_ratio": float(text_to_image_ratio),
            "comprehension_speed": speed
        }
    
    def cognitive_dissonance_index(self,
                                  frames_rgb: List[np.ndarray],
                                  vision_feats: Dict[str, Any],
                                  clip_model,
                                  tokenize,
                                  backend: str,
                                  rep_indices: List[int]) -> Dict[str, Any]:
        """
        Detects contradictions between visual elements.
        
        Approach:
        1. Semantic consistency: do CLIP embeddings of different regions align?
        2. Color-brand mismatch: luxury brands shouldn't have garish colors
        3. Quality signals: detect low-quality overlays on high-quality imagery
        """
        if len(frames_rgb) == 0 or len(rep_indices) == 0:
            return {
                "dissonance_index": 0.0,
                "semantic_consistency": 1.0,
                "color_brand_alignment": 1.0,
                "quality_consistency": 1.0
            }
        
        first_frame = frames_rgb[rep_indices[0]]
        h, w = first_frame.shape[:2]
        
        # 1. Semantic Consistency
        # Split frame into quadrants and compare CLIP embeddings
        quads = [
            first_frame[:h//2, :w//2],      # top-left
            first_frame[:h//2, w//2:],      # top-right
            first_frame[h//2:, :w//2],      # bottom-left
            first_frame[h//2:, w//2:]       # bottom-right
        ]
        
        quad_embeddings = []
        from PIL import Image
        for quad in quads:
            if quad.size == 0:
                continue
            try:
                # Use the same CLIP preprocessing from vision.py logic
                pil_quad = Image.fromarray(quad)
                # We need access to clip_preprocess - let's approximate
                # Resize to 224x224 and normalize
                quad_resized = cv2.resize(quad, (224, 224))
                quad_tensor = torch.from_numpy(quad_resized).float().permute(2, 0, 1) / 255.0
                # Normalize with ImageNet stats
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                quad_tensor = (quad_tensor - mean) / std
                quad_tensor = quad_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    emb = clip_model.encode_image(quad_tensor)
                    if backend == "open_clip":
                        emb = F.normalize(emb, dim=-1)
                    else:
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                quad_embeddings.append(emb[0].cpu().numpy())
            except Exception:
                continue
        
        # Compute pairwise cosine similarities
        if len(quad_embeddings) >= 2:
            similarities = []
            for i in range(len(quad_embeddings)):
                for j in range(i+1, len(quad_embeddings)):
                    sim = np.dot(quad_embeddings[i], quad_embeddings[j])
                    similarities.append(sim)
            semantic_consistency = safe_mean(similarities)
        else:
            semantic_consistency = 1.0  # assume consistent if we can't measure
        
        # 2. Color-Brand Alignment
        # Check if color palette matches vertical/brand positioning
        warmth = vision_feats.get("warmth", 1.0)
        vertical = vision_feats.get("vertical_top1", "")
        
        # Heuristic rules
        color_brand_alignment = 1.0
        
        # Luxury brands should have: high contrast, moderate warmth, low clutter
        if "automotive" in vertical.lower() or "finance" in vertical.lower():
            contrast = vision_feats.get("contrast", 0.5)
            clutter = vision_feats.get("clutter", 0.5)
            # Expect high contrast, low clutter
            expected_quality = contrast * (1.0 - clutter)
            if expected_quality < 0.3:
                color_brand_alignment *= 0.7  # penalize
        
        # Beauty/skincare should be warm
        if "beauty" in vertical.lower() or "skincare" in vertical.lower():
            if warmth < 0.8:  # too cool
                color_brand_alignment *= 0.8
        
        # Fitness should have high energy (high warmth OR high contrast)
        if "fitness" in vertical.lower():
            contrast = vision_feats.get("contrast", 0.5)
            energy = max(warmth, contrast)
            if energy < 0.5:
                color_brand_alignment *= 0.8
        
        # 3. Quality Consistency
        # Check for low-quality overlays on high-quality base
        # Proxy: high logo_stability + high clutter = bad overlay
        logo_stability = vision_feats.get("logo_stability", 0.0)
        clutter = vision_feats.get("clutter", 0.5)
        contrast = vision_feats.get("contrast", 0.5)
        
        # High clutter + low contrast suggests messy overlay
        if clutter > 0.6 and contrast < 0.4:
            quality_consistency = 0.6
        else:
            quality_consistency = 1.0
        
        # Final dissonance index (0-1, higher = more dissonance)
        dissonance_index = 1.0 - (
            semantic_consistency * 0.5 +
            color_brand_alignment * 0.3 +
            quality_consistency * 0.2
        )
        
        return {
            "dissonance_index": float(dissonance_index),
            "semantic_consistency": float(semantic_consistency),
            "color_brand_alignment": float(color_brand_alignment),
            "quality_consistency": float(quality_consistency)
        }
    
    def visual_path_entropy(self,
                           frames_rgb: List[np.ndarray],
                           rep_indices: List[int]) -> Dict[str, Any]:
        """
        Measures how scattered vs. focused the visual attention path is.
        
        Uses saliency maps to predict gaze patterns.
        High entropy = scattered attention = poor performance
        Low entropy = clear focal point = good performance
        """
        if len(frames_rgb) == 0 or len(rep_indices) == 0:
            return {
                "path_entropy": 0.0,
                "attention_focus": "unknown",
                "primary_focal_strength": 0.0,
                "attention_distribution": "uniform"
            }
        
        saliency_maps = []
        
        for idx in rep_indices:
            if idx < 0 or idx >= len(frames_rgb):
                continue
            
            frame = frames_rgb[idx]
            
            # Generate saliency map
            success, saliency_map = self.saliency.computeSaliency(frame)
            
            if not success or saliency_map is None:
                continue
            
            # Normalize to 0-255
            saliency_map = (saliency_map * 255).astype(np.uint8)
            saliency_maps.append(saliency_map)
        
        if len(saliency_maps) == 0:
            return {
                "path_entropy": 0.0,
                "attention_focus": "unknown",
                "primary_focal_strength": 0.0,
                "attention_distribution": "uniform"
            }
        
        # Average saliency across representative frames
        avg_saliency = np.mean(np.stack(saliency_maps), axis=0)
        
        # 1. Compute entropy of saliency distribution
        # Treat saliency map as probability distribution
        saliency_flat = avg_saliency.flatten()
        saliency_flat = saliency_flat / (np.sum(saliency_flat) + 1e-9)
        
        # Remove zeros for entropy calculation
        saliency_nonzero = saliency_flat[saliency_flat > 1e-9]
        path_entropy_value = scipy_entropy(saliency_nonzero)
        
        # Normalize entropy (max entropy for uniform distribution)
        max_entropy = np.log(len(saliency_nonzero)) if len(saliency_nonzero) > 0 else 1.0
        normalized_entropy = path_entropy_value / (max_entropy + 1e-9)
        
        # 2. Primary focal strength
        # Measure how strong the brightest region is vs. the rest
        max_saliency = np.max(avg_saliency)
        mean_saliency = np.mean(avg_saliency)
        focal_strength = (max_saliency - mean_saliency) / (max_saliency + 1e-9)
        
        # 3. Attention distribution pattern
        # Count significant "hot spots"
        threshold = np.percentile(avg_saliency, 90)  # top 10%
        hot_spots = avg_saliency > threshold
        
        # Label connected components
        num_labels, labels = cv2.connectedComponents(hot_spots.astype(np.uint8))
        num_hotspots = num_labels - 1  # subtract background
        
        if num_hotspots <= 1:
            distribution = "focused"  # single focal point
        elif num_hotspots <= 3:
            distribution = "structured"  # few clear points
        else:
            distribution = "scattered"  # many competing points
        
        # 4. Overall attention focus category
        if normalized_entropy < 0.3 and focal_strength > 0.6:
            focus = "strong"  # clear hierarchy
        elif normalized_entropy < 0.5 and focal_strength > 0.4:
            focus = "moderate"  # some structure
        else:
            focus = "weak"  # scattered
        
        return {
            "path_entropy": float(normalized_entropy),
            "attention_focus": focus,
            "primary_focal_strength": float(focal_strength),
            "attention_distribution": distribution,
            "num_attention_hotspots": int(num_hotspots)
        }
    
    def extract_cognitive_features(self,
                                  frames_rgb: List[np.ndarray],
                                  vision_feats: Dict[str, Any],
                                  clip_model,
                                  tokenize,
                                  backend: str,
                                  rep_indices: List[int]) -> Dict[str, Any]:
        """
        High-level wrapper to extract all cognitive features.
        """
        comprehension = self.immediate_comprehension_score(
            frames_rgb, vision_feats, rep_indices
        )
        
        dissonance = self.cognitive_dissonance_index(
            frames_rgb, vision_feats, clip_model, tokenize, backend, rep_indices
        )
        
        attention = self.visual_path_entropy(
            frames_rgb, rep_indices
        )
        
        return {
            **comprehension,
            **dissonance,
            **attention
        }