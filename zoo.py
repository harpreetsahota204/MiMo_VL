
import os
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.utils import is_flash_attn_2_available

from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in detecting and localizating meaningful visual elements. 

You can detect and localize objects, components, people, places, things, and UI elements in images using 2D bound boxes.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "detections": [
        {
            "bbox_2d": [x1, y1, x2, y2],
            "label": "descriptive label for the bounding box"
        },
        {
            "bbox_2d": [x1, y1, x2, y2],
            "label": "descriptive label for the bounding box"
        }
    ]
}
```

The JSON should contain bounding boxes in pixel coordinates [x1,y1,x2,y2] format, where:
- x1,y1 is the top-left corner
- x2,y2 is the bottom-right corner
- Provide specific, descriptive labels for each detected element
- Include all relevant elements that match the user's request
- For UI elements, include their function when possible (e.g., "Login Button" rather than just "Button")
- If many similar elements exist, prioritize the most prominent or relevant ones

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and detect.
"""

DEFAULT_KEYPOINT_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in key point detection across any visual domain. A key point represents the center of any meaningful visual element. 

Key points should adapt to the context (physical world, digital interfaces, UI elements, etc.) while maintaining consistent accuracy and relevance. 

For each key point identify the key point and provide a contextually appropriate label and always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "keypoints": [
        {
            "point_2d": [x, y],
            "label": "descriptive label for the point"
        }
    ]
}
```

The JSON should contain points in pixel coordinates [x,y] format, where:
- x is the horizontal center coordinate of the visual element
- y is the vertical center coordinate of the visual element
- Include all relevant elements that match the user's request
- You can point to multiple visual elements

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and point.
"""

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are a helpful assistant. You specializes in comprehensive classification across any visual domain, capable of analyzing:

Unless specifically requested for single-class output, multiple relevant classifications can be provided.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "classifications": [
        {
            "label": "descriptive class label"
        }
    ]
}
```

The JSON should contain a list of classifications where:
- Each classification must have a 'label' field
- Labels should be descriptive strings describing what you've identified in the image
- The response should be a list of classifications
"""
   
DEFAULT_OCR_SYSTEM_PROMPT = """You are a helpful assistant specializing in text detection and recognition (OCR) in images. Your can read, detect, and locate text from any visual content, including documents, UI elements, signs, or any other text-containing regions.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "text_detections": [
        {
            "bbox_2d": [x1, y1, x2, y2],  // Coordinates: [top-left x, top-left y, bottom-right x, bottom-right y]
            "text_type": "title|abstract|heading|paragraph|button|link|label|title|menu_item|input_field|icon|list_item|etc.",  // Select appropriate text category
            "text": "Exact text content found in this region"  // Transcribe text exactly as it appears
        }
    ]
}
```

The JSON should contain bounding boxes in pixel coordinates [x1,y1,x2,y2] format, where:
- x1,y1 is the top-left corner
- x2,y2 is the bottom-right corner
- 'text_type' is important to get right, it's the text region category based on the document, including but not limited to: title, abstract, heading, paragraph, button, link, label, icon, menu item, etc.
- The 'text' field should be a string containing the exact text content found in the region

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's perform the OCR detections.
"""

DEFAULT_VQA_SYSTEM_PROMPT = "You are a helpful assistant. You provide clear and concise answerss to questions about images. Report answers in natural language text in English."

DEFAULT_AGENTIC_PROMPT = """You are a GUI agent. You interact with graphical user interfaces. Your goal is to accomplish user requests by analyzing user interface and generating an appropriate action.

Your action space consists of the following:

- click: {"action":"click","point_2d":[x,y],"text":text} – Click at (x,y) on an element with the given text.

- scroll: {"action":"scroll", "point_2d":[x,y], "direction":dir,"scroll_distance":dist} – Scroll in the given direction by the specified distance.

- input: {"action":"input","point_2d":[x,y],"text":text} – Type the specified text at coordinates (x,y).

- drag: {"action":"drag","point_2d":[x,y],"end_point":[x2,y2]} – Drag from start_point to end_point.

- open: {"action":"open","point_2d":[x,y],"app":app_name} – Open the specified application.

- press: {"action":"press", "point_2d":[x,y]} – Press the specified hotkeys.

- finished: {"action":"finished","point_2d":[x,y], "status":status} – Mark the task as complete.

- longpress: {"action":"longpress", "point_2d":[x,y]} – Long press at (x,y).

- hover: {"action":"hover", "point_2d":[x,y]} – Hover the mouse over a location.

- select: {"action":"select","point_2d":[x,y],"text":text} – Select the specified text.

- wait: {"action":"wait", "point_2d":[x,y]} – Pause for a brief moment.

- appswitch: {"action":"appswitch","point_2d":[x,y], "app":app_name} – Switch to the specified application.

Analyze the user interface and determine the appropriate action. Always return your response as valid JSON wrapped in ```json blocks, following this structure:

```json
{
    "keypoints": [
        {
            "point_2d": [x, y],
            "action": {
                "action": "click|scroll|input|drag|open|press|finished|longpress|hover|select|wait|appswitch",
                // Include only the relevant fields for the chosen action:
                "text": "text for click, input, or select actions",
                "direction": "direction for scroll actions",
                "scroll_distance": value for scroll actions,
                "end_point": [x2, y2] for drag actions,
                "app": "application name for open or appswitch actions",
                "status": "status for finished actions"
            }
        }
    ]
}
```
"""

MIMOVL_OPERATIONS = {
    "detect": DEFAULT_DETECTION_SYSTEM_PROMPT,
    "point": DEFAULT_KEYPOINT_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
    "ocr": DEFAULT_OCR_SYSTEM_PROMPT,
    "agentic": DEFAULT_AGENTIC_PROMPT,
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class MimoVLModel(SamplesMixin, Model):
    """A FiftyOne model for running MimoVL vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        model_kwargs = {
            "device_map":self.device,
            }

        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Only set specific torch_dtype for CUDA devices
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
            )

        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )

        self.model.eval()

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    
    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in MIMOVL_OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(MIMOVL_OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else MIMOVL_OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output and extract reasoning information.
        
        This method handles multiple formats of model output:
        1. Text with <think> tags containing reasoning
        2. JSON within markdown code blocks (```json)
        3. Raw JSON strings
        
        Args:
            s: Raw string output from the model containing JSON and possibly reasoning
                
        Returns:
            Dict: Dictionary containing:
                - data: The parsed JSON content
                - reasoning: Text extracted from <think> tags
            None: If parsing fails
            Original input: If input is not a string
        
        Example input:
            "<think>This image contains a person</think>
            ```json
            {"detections": [{"bbox": [0,0,100,100], "label": "person"}]}
            ```"
        """
        # Return non-string inputs as-is
        if not isinstance(s, str):
            return s
        
        # Extract reasoning from between <think> tags if present
        reasoning = ""
        if "<think>" in s and "</think>" in s:
            try:
                # Split on tags and take content between them
                reasoning = s.split("<think>")[1].split("</think>")[0].strip()
            except:
                logger.debug("Failed to extract reasoning from <think> tags")
        
        # Extract JSON content from markdown code blocks if present
        if "```json" in s:
            try:
                # Split on markdown markers and take JSON content
                json_str = s.split("```json")[1].split("```")[0].strip()
            except:
                json_str = s
        else:
            json_str = s
            
        # Attempt to parse the JSON string
        try:
            parsed_json = json.loads(json_str)
            return {
                "data": parsed_json,  # The actual JSON content
                "reasoning": reasoning  # The extracted reasoning text
            }
        except:
            # Log parsing failures for debugging
            logger.debug(f"Failed to parse JSON: {json_str[:200]}")
            return None

    def _to_detections(self, boxes: List[Dict], image_width: int, image_height: int) -> fo.Detections:
        """Convert bounding boxes to FiftyOne Detections with associated reasoning.
        
        Takes detection results and converts them to FiftyOne's format, including:
        - Coordinate normalization
        - Label extraction
        - Reasoning attachment
        
        Args:
            boxes: Detection results, either:
                - List of detection dictionaries
                - Dictionary containing 'data' and 'reasoning'
            image_width: Original image width in pixels
            image_height: Original image height in pixels
        
        Returns:
            fo.Detections: FiftyOne Detections object containing all converted detections
            
        Example input:
            {
                "data": [{"bbox": [0,0,100,100], "label": "person"}],
                "reasoning": "Detected a person in the upper left"
            }
        """
        detections = []
        
        # Extract reasoning if present in dictionary format
        reasoning = boxes.get("reasoning", "") if isinstance(boxes, dict) else ""
        
        # Handle nested dictionary structures
        if isinstance(boxes, dict):
            # Try to get data field, fall back to original dict if not found
            boxes = boxes.get("data", boxes)
            if isinstance(boxes, dict):
                # If still a dict, try to find first list value
                boxes = next((v for v in boxes.values() if isinstance(v, list)), boxes)
        
        # Ensure we're working with a list of boxes
        boxes = boxes if isinstance(boxes, list) else [boxes]
        
        # Process each bounding box
        for box in boxes:
            try:
                # Extract bbox coordinates, checking both possible keys
                bbox = box.get('bbox_2d', box.get('bbox', None))
                if not bbox:
                    continue
                    
                # Convert pixel coordinates to normalized [0,1] coordinates
                x1, y1, x2, y2 = map(float, bbox)
                x = x1 / image_width  # Left coordinate
                y = y1 / image_height  # Top coordinate
                w = (x2 - x1) / image_width  # Width
                h = (y2 - y1) / image_height  # Height
                
                # Create FiftyOne Detection object
                detection = fo.Detection(
                    label=str(box.get("label", "object")),
                    bounding_box=[x, y, w, h],
                    reasoning=reasoning  # Attach reasoning to detection
                )
                detections.append(detection)
                    
            except Exception as e:
                logger.debug(f"Error processing box {box}: {e}")
                continue
                    
        return fo.Detections(detections=detections)

    def _to_ocr_detections(self, boxes: List[Dict], image_width: int, image_height: int) -> fo.Detections:
        """Convert OCR results to FiftyOne Detections with reasoning.
        
        Takes OCR detection results and converts them to FiftyOne's format, including:
        - Coordinate normalization
        - Text content preservation
        - Text type categorization
        - Reasoning attachment
        
        Args:
            boxes: OCR detection results, either:
                - List of OCR dictionaries
                - Dictionary containing 'data' and 'reasoning'
            image_width: Original image width in pixels
            image_height: Original image height in pixels
        
        Returns:
            fo.Detections: FiftyOne Detections object containing all converted OCR detections
            
        Example input:
            {
                "data": {
                    "text_detections": [
                        {
                            "bbox": [0,0,100,100],
                            "text": "Hello",
                            "text_type": "heading"
                        }
                    ]
                },
                "reasoning": "Found heading text in top-left corner"
            }
        """
        detections = []
        
        # Extract reasoning if present in dictionary format
        reasoning = boxes.get("reasoning", "") if isinstance(boxes, dict) else ""
        
        # Handle nested dictionary structures
        if isinstance(boxes, dict):
            # Try to get data field, fall back to original dict if not found
            boxes = boxes.get("data", boxes)
            if isinstance(boxes, dict):
                # If still a dict, try to find first list value (usually "text_detections")
                boxes = next((v for v in boxes.values() if isinstance(v, list)), boxes)
        
        # Ensure boxes is a list, even for single box input
        boxes = boxes if isinstance(boxes, list) else [boxes]
        
        # Process each OCR box
        for box in boxes:
            try:
                # Extract bbox coordinates, checking both possible keys
                bbox = box.get('bbox_2d', box.get('bbox', None))
                if not bbox:
                    continue
                    
                # Extract text content and type
                text = box.get('text')
                text_type = box.get('text_type', 'text')  # Default to 'text' if not specified
                
                # Skip if no text content
                if not text:
                    continue
                    
                # Convert pixel coordinates to normalized [0,1] coordinates
                x1, y1, x2, y2 = map(float, bbox)
                x = x1 / image_width  # Left coordinate
                y = y1 / image_height  # Top coordinate
                w = (x2 - x1) / image_width  # Width
                h = (y2 - y1) / image_height  # Height
                
                # Create FiftyOne Detection object
                detection = fo.Detection(
                    label=str(text_type),
                    bounding_box=[x, y, w, h],
                    text=str(text),
                    reasoning=reasoning  # Attach reasoning to detection
                )
                detections.append(detection)
                    
            except Exception as e:
                logger.debug(f"Error processing OCR box {box}: {e}")
                continue
                    
        return fo.Detections(detections=detections)

    def _to_keypoints(self, points: List[Dict], image_width: int, image_height: int) -> fo.Keypoints:
        """Convert keypoint detections to FiftyOne Keypoints with reasoning.
        
        Processes keypoint coordinates and normalizes them to [0,1] range while
        preserving associated labels and reasoning.
        
        Args:
            points: Keypoint detection results, either:
                - List of keypoint dictionaries
                - Dictionary containing 'data' and 'reasoning'
            image_width: Original image width in pixels
            image_height: Original image height in pixels
            
        Returns:
            fo.Keypoints: FiftyOne Keypoints object containing all converted keypoints
            
        Example input:
            {
                "data": [{"point_2d": [100,100], "label": "nose"}],
                "reasoning": "Identified facial features"
            }
        """
        keypoints = []
        
        # Extract reasoning if present
        reasoning = points.get("reasoning", "") if isinstance(points, dict) else ""
        
        # Handle nested dictionary structures
        if isinstance(points, dict):
            points = points.get("data", points)
            if isinstance(points, dict):
                points = next((v for v in points.values() if isinstance(v, list)), points)
        
        # Process each keypoint
        for point in points:
            try:
                # Extract and normalize coordinates
                x, y = point["point_2d"]
                # Handle tensor inputs if present
                x = float(x.cpu() if torch.is_tensor(x) else x)
                y = float(y.cpu() if torch.is_tensor(y) else y)
                
                # Normalize coordinates to [0,1] range
                normalized_point = [
                    x / image_width,
                    y / image_height
                ]
                
                # Create FiftyOne Keypoint object
                keypoint = fo.Keypoint(
                    label=str(point.get("label", "point")),
                    points=[normalized_point],
                    reasoning=reasoning
                )
                keypoints.append(keypoint)
            except Exception as e:
                logger.debug(f"Error processing point {point}: {e}")
                continue
                
        return fo.Keypoints(keypoints=keypoints)
    
    def _to_agentic_keypoints(self, actions: Dict, image_width: int, image_height: int) -> fo.Keypoints:
        """Convert agentic actions to FiftyOne Keypoints.
        
        Args:
            actions: Dictionary containing keypoints with point_2d and action JSON string
            image_width: Original image width in pixels
            image_height: Original image height in pixels
        """
        keypoints = []
        
        # Handle nested dictionary structures
        if isinstance(actions, dict):
            actions = actions.get("data", actions)
            if isinstance(actions, dict):
                actions = next((v for v in actions.values() if isinstance(v, list)), actions)
        
        for idx, kp in enumerate(actions):
            try:
                # Extract and normalize point coordinates
                x, y = map(float, kp["point_2d"])
                point = [x / image_width, y / image_height]
                
                # Parse the action JSON string into a dict
                action_data = json.loads(kp["action"]) if isinstance(kp["action"], str) else kp["action"]
                action_type = action_data["action"]
                
                # Base metadata with sequence index and action type
                metadata = {
                    "sequence_idx": idx,
                    "action": action_type
                }
                
                # Add action-specific metadata
                if action_type in ["click", "input", "select"]:
                    metadata["text"] = action_data.get("text")
                    
                elif action_type == "scroll":
                    metadata["direction"] = action_data.get("direction")
                    metadata["scroll_distance"] = action_data.get("scroll_distance")
                    
                elif action_type == "drag":
                    if "end_point" in action_data:
                        x2, y2 = map(float, action_data["end_point"])
                        metadata["end_point"] = [x2 / image_width, y2 / image_height]
                        
                elif action_type in ["open", "appswitch"]:
                    metadata["app"] = action_data.get("app")
                    
                elif action_type == "finished":
                    metadata["status"] = action_data.get("status")
                    
                # No additional parameters needed for these actions
                elif action_type in ["press", "longpress", "hover", "wait"]:
                    pass
                
                keypoint = fo.Keypoint(
                    label=action_type,
                    points=[point],
                    metadata=metadata
                )
                keypoints.append(keypoint)
                    
            except Exception as e:
                logger.debug(f"Error processing keypoint {kp}: {e}")
                continue
                    
        return fo.Keypoints(keypoints=keypoints)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert classification results to FiftyOne Classifications with reasoning.
        
        Processes classification labels and associated reasoning into FiftyOne's format.
        
        Args:
            classes: Classification results, either:
                - List of classification dictionaries
                - Dictionary containing 'data' and 'reasoning'
                
        Returns:
            fo.Classifications: FiftyOne Classifications object containing all results
            
        Example input:
            {
                "data": [{"label": "cat"}, {"label": "animal"}],
                "reasoning": "Image shows a domestic cat"
            }
        """
        classifications = []
        
        # Extract reasoning if present
        reasoning = classes.get("reasoning", "") if isinstance(classes, dict) else ""
        
        # Handle nested dictionary structures
        if isinstance(classes, dict):
            classes = classes.get("data", classes)
            if isinstance(classes, dict):
                classes = next((v for v in classes.values() if isinstance(v, list)), classes)
        
        # Process each classification
        for cls in classes:
            try:
                # Create FiftyOne Classification object
                classification = fo.Classification(
                    label=str(cls["label"]),
                    reasoning=reasoning  # Attach reasoning to classification
                )
                classifications.append(classification)
            except Exception as e:
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        return fo.Classifications(classifications=classifications)

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Keypoints, fo.Classifications, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/points/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance
        
        if not prompt:
            raise ValueError("No prompt provided.")
        
        messages = [
            {
                "role": "system", 
                "content": [  
                    {
                        "type": "text",
                        "text": self.system_prompt
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image", 
                        "image": sample.filepath if sample else image
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
            )
        
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text], 
            images=image_inputs,
            videos=video_inputs,
            padding=True, 
            return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                temperature=0.1, 
                top_p=0.95, 
                do_sample=True, 
                max_new_tokens=16384,
                pad_token_id=self.processor.tokenizer.eos_token_id
                )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Get image dimensions and convert to float
        input_height = float(inputs['image_grid_thw'][0][1].cpu() * 14)
        input_width = float(inputs['image_grid_thw'][0][2].cpu() * 14)

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()
        elif self.operation == "detect":
            parsed_output = self._parse_json(output_text)
            return self._to_detections(parsed_output, input_width, input_height)
        elif self.operation == "ocr":
            parsed_output = self._parse_json(output_text)
            return self._to_ocr_detections(parsed_output, input_width, input_height)
        elif self.operation == "point":
            parsed_output = self._parse_json(output_text)
            return self._to_keypoints(parsed_output, input_width, input_height)
        elif self.operation == "classify":
            parsed_output = self._parse_json(output_text)
            return self._to_classifications(parsed_output)
        elif self.operation == "agentic":
            parsed_output = self._parse_json(output_text)
            return self._to_agentic_keypoints(parsed_output, input_width, input_height)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)
