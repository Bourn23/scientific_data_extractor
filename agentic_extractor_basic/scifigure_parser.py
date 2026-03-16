import os
import json
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
import re

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal

class ValueRange(BaseModel):
    min: float = Field(..., description="Minimum value on this axis")
    max: float = Field(..., description="Maximum value on this axis")

class AxisDetails(BaseModel):
    title: str = Field(..., description="The full text label of the axis (e.g., 'Ionic Conductivity', '1000/T')")
    unit: Optional[str] = Field(None, description="The specific unit extracted from the label (e.g., 'S/cm', 'K⁻¹', 'eV')")

    quantity_type: Literal[
        "conductivity", 
        "activation_energy", 
        "temperature_absolute",   # Normal T (Celsius/Kelvin)
        "temperature_inverse",    # Arrhenius 1000/T
        "stoichiometry",          # x, composition, doping amount
        "frequency",              # NEW: Add this explicitly
        "impedance",              # NEW: For Nyquist plots
        "voltage", 
        "capacity",
        "other"
    ] = Field(..., description="The physical quantity being plotted. Classify '1000/T' as 'temperature_inverse', 'Hz/frequency' as 'frequency'.")

    scale_type: Literal["linear", "log", "reciprocal", "unknown"] = Field(
        "linear", description="The scale of the axis. 'log' for log(σ), 'reciprocal' for 1000/T."
    )
    
    # NEW: Add value range for validation
    value_range: Optional[ValueRange] = Field(
        None, 
        description="Min and max values on this axis"
    )
class SubplotDetection(BaseModel):
    contains_conductivity_data: bool = Field(
        ..., 
        description="TRUE only if this subplot plots Ionic Conductivity (σ) vs Temperature or Stoichiometry. FALSE for frequency plots, Nyquist plots, XRD, or structure diagrams."
    )

    plot_type: Literal[
        "arrhenius",           # σ vs 1000/T
        "temperature_dep",     # σ vs T
        "composition_dep",     # σ vs x (stoichiometry)
        "frequency_dep",       # σ vs f (impedance spectroscopy)
        "nyquist",            # Z'' vs Z'
        "other"
    ] = Field(
        "other", 
        description="Classify the plot type to aid filtering"
    )

    # CRITICAL: Separate axes to handle Figure 6 (Dual Axis) scenarios
    x_axis: AxisDetails
    left_y_axis: AxisDetails
    right_y_axis: Optional[AxisDetails] = Field(None, description="Only populated if there is a distinct secondary Y-axis on the right side.")
    
    box_2d: List[int] = Field(..., description="Bounding box [ymin, xmin, ymax, xmax] in 0-1000 integer coordinates.")
    label: str = Field(..., description="Panel label if present (e.g., 'a', 'b'). Use 'main' if single plot.")
    legend_labels: Optional[List[str]] = Field(
        None,
        description="List of series labels from the legend (e.g., ['223K', '263K', '303K'] or ['Fe doped', 'Al doped'])"
    )
class FigureAnalysis(BaseModel):
    is_multi_panel: bool = Field(..., description="True if image contains multiple subplots")
    subplots: List[SubplotDetection]   

class DataPoint(BaseModel):
    series_label: str = Field(..., description="Name from the legend (e.g. 'Sintered', 'Cold-pressed')")
    x_value: float = Field(..., description="Numeric X value")
    y_value: float = Field(..., description="Numeric Y value")
    
    # CRITICAL: Explicitly link point to the correct axis definition index
    mapped_y_axis: Literal["left", "right"] = Field(
        "left", 
        description="Does this point belong to the 'Left' Y-axis or the 'Right' Y-axis defined in the prompt?"
    )

class DataSeries(BaseModel):
    series_label: str = Field(..., description="Name from the legend (e.g. 'Sintered', 'Cold-pressed')")
    
    # VECTORIZED DATA
    x_values: List[float] = Field(..., description="List of X-coordinates for this series.")
    y_values: List[float] = Field(..., description="List of Y-coordinates corresponding to the X-values.")
    
    # ROUTING (Applied to the whole list)
    mapped_y_axis: Literal["left", "right"] = Field(
        "left", 
        description="Which Y-axis definition (from the prompt) applies to this entire series?"
    )
class ExtractionResult(BaseModel):
    data_series: List[DataSeries] = Field(..., description="Grouped data extracted from the plot.")
    summary: str = Field(..., description="One sentence summary of the trend (e.g. 'Conductivity peaks at x=0.4').")
    annotated_temperature: Optional[str] = Field(
        None,
        description="If NEITHER axis is temperature but a single temperature is annotated inside the plot "
                    "(e.g., 'at 70°C', 'T = 300K', 'room temperature'), extract it here as a string "
                    "(e.g., '70°C', '300K'). Null if no annotation found or if an axis already represents temperature."
    )


class SciFigureParser:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", debug: bool = False, save_debug: bool = True):
        self.api_key = api_key
        self.model_name = model_name
        self.debug = debug
        self.save_debug = save_debug
        self.client_v1 = genai.Client(api_key=api_key)
        self.client_alpha = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    def _is_gemini_3(self) -> bool:
        return "gemini-3" in self.model_name

    def _get_client(self):
        return self.client_alpha if self._is_gemini_3() else self.client_v1

    def _get_debug_path(self, image_path: str, suffix: str, ext: Optional[str] = None) -> str:
        """
        Creates a 'scifig_debug' folder in the image_path directory and returns the debug path.
        Avoids nested 'scifig_debug' folders.
        """
        dir_name = os.path.abspath(os.path.dirname(image_path))
        base_name = os.path.basename(image_path)
        file_name, original_ext = os.path.splitext(base_name)
        
        if os.path.basename(dir_name) == "scifig_debug":
            debug_dir = dir_name
        else:
            debug_dir = os.path.join(dir_name, "scifig_debug")
            
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)
            
        extension = ext if ext else original_ext
        if not extension.startswith('.'):
            extension = f".{extension}"
            
        return os.path.join(debug_dir, f"{file_name}{suffix}{extension}")

    def _sanitize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively removes 'additionalProperties' from the schema.
        Also handles other potential Gemini API unsupported schema features if needed.
        """
        if not isinstance(schema, dict):
            return schema
            
        # Create a copy to avoid modifying the original in place if that matters, 
        # though usually we pass a fresh one.
        new_schema = schema.copy()
        
        if "additionalProperties" in new_schema:
            del new_schema["additionalProperties"]
            
        # Recursive cleaning
        for key, value in new_schema.items():
            if isinstance(value, dict):
                new_schema[key] = self._sanitize_schema(value)
            elif isinstance(value, list):
                new_schema[key] = [self._sanitize_schema(item) if isinstance(item, dict) else item for item in value]
                
        return new_schema

    def _get_image_part(self, data: bytes, mime_type: str = "image/jpeg") -> types.Part:
        if self._is_gemini_3():
            return types.Part(
                inline_data=types.Blob(
                    mime_type=mime_type,
                    data=data,
                ),
                media_resolution={"level": "media_resolution_high"}
            )
        else:
            return types.Part.from_bytes(data=data, mime_type=mime_type)

    def detect_subplot(self, image_path: str, query: str) -> Dict[str, Any]:
        """
        Locates a subplot in a multi-panel figure based on a query.
        Returns normalized coordinates: ymin, xmin, ymax, xmax (0-1000).
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        prompt = f"""Identify and locate all subplots/panels related to "{query}" in this figure. 
        
        CRITICAL INSTRUCTIONS: 
        1. Capture the ENTIRE subplot including all axes, axis titles/labels, units, tick marks, and LEGEND.
        2. LEGEND IMPORTANCE: The bounding box MUST include the legend/key (which explains symbols/colors), even if it is floating inside the plot or outside the axes.
        3. Specifically for Axis Titles: Ensure the bounding box extends far enough to include the text describing the units (e.g., 'S/cm', '1000/T').
        4. Identify if this is a multi-panel figure.
        5. For EVERY relevant subplot:
           - Provide a bounding box.
           - Assign a short label (e.g., 'A', 'B').
           - Extract the text of the X and Y axis titles including units.
        6. If there is only ONE plot total, set "is_multi_plot" to false.
        7. Return the result in a structured JSON format."""

        response = self._get_client().models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    parts=[
                        types.Part(text=prompt),
                        self._get_image_part(image_data)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "is_multi_plot": {"type": "BOOLEAN", "description": "True if the figure contains multiple distinct subplots or panels."},
                        "detections": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "ymin": {"type": "NUMBER"},
                                    "xmin": {"type": "NUMBER"},
                                    "ymax": {"type": "NUMBER"},
                                    "xmax": {"type": "NUMBER"},
                                    "label": {"type": "STRING", "description": "Label for this specific subplot."},
                                    "xAxisTitle": {"type": "STRING", "description": "The detected text for the X-axis label and unit."},
                                    "yAxisTitle": {"type": "STRING", "description": "The detected text for the primary Y-axis label and unit."},
                                    "secondaryYAxisTitle": {"type": "STRING", "description": "The detected text for the secondary Y-axis label and unit if it exists."}
                                },
                                "required": ["ymin", "xmin", "ymax", "xmax", "xAxisTitle", "yAxisTitle"]
                            }
                        },
                        "isIonicConductivity": {"type": "BOOLEAN", "description": "True if the figure contains ionic conductivity measurements."}
                    },
                    "required": ["is_multi_plot", "detections", "isIonicConductivity"]
                }
            )
        )
        
        result = json.loads(response.text)

        # Post-detection: Normalize axis titles to identify stoichiometry or temperature
        for det in result.get('detections', []):
            x_title = det.get('xAxisTitle', '').lower()
            if any(marker in x_title for marker in ['x=', 'composition', 'stoichiometry', 'substitution', 'doped', 'amount']) or x_title.strip() == 'x':
                 det['xAxisType'] = 'stoichiometry'
            elif any(marker in x_title for marker in ['t ', 'temp', '1000/t', 'k-1', '°c', 'k ']):
                 det['xAxisType'] = 'temperature'
            else:
                 det['xAxisType'] = 'unknown'

        # Inject Usage Metadata
        if response.usage_metadata:
            result['_usage_metadata'] = {
                'prompt_token_count': response.usage_metadata.prompt_token_count,
                'candidates_token_count': response.usage_metadata.candidates_token_count,
                'total_token_count': response.usage_metadata.total_token_count
            }

        if self.debug and self.save_debug:
            self._visualize_detection(image_path, result.get('detections', []), query)

        return result

    async def detect_subplot_async(self, image_path: str, query: str) -> Dict[str, Any]:
        """
        Locates a subplot in a multi-panel figure based on a query (Asynchronous).
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        prompt = f"""Analyze this scientific figure to locate "{query}" plots/subplots.
        
        CRITICAL FILTERING INSTRUCTIONS:
        1. IDENTIFY X-AXIS TYPE carefully:
        - If X-axis shows "f", "Hz", "frequency" → classify as 'other' and set contains_conductivity_data=FALSE
        - If X-axis shows "1000/T", "T⁻¹", "K⁻¹" → classify as 'temperature_inverse'
        - If X-axis shows "T", "Temperature", "°C", "K" → classify as 'temperature_absolute'
        - If X-axis shows "x", "composition", "stoichiometry", "Li content" → classify as 'stoichiometry'
        
        2. LEGEND INTERPRETATION:
        - In conductivity-vs-frequency plots: Legend labels like "223K", "263K" are TEMPERATURES (experimental conditions), NOT composition
        - In Arrhenius plots: Legend labels describe material composition (e.g., "Fe doped", "x=0.2")
        - This distinction is CRITICAL for proper data extraction
        
        3. SET contains_conductivity_data=TRUE only if:
        - Y-axis shows conductivity (σ, S/cm, log(σ))
        - AND X-axis is temperature-related OR stoichiometry
        - REJECT: frequency plots, Nyquist plots, XRD, structure diagrams
        
        4. DUAL Y-AXES:
        - Some plots show conductivity (S/cm) on one axis and activation energy (eV) on the other
        - Extract BOTH left_y_axis and right_y_axis separately with their units
        
        5. BOUNDING BOX:
        - Include ALL axis labels, tick marks, and LEGEND
        - Extend box to capture unit labels (e.g., "S cm⁻¹", "1000/T")
        
        6. UNITS EXTRACTION:
        - Parse axis labels to extract explicit units
        - Example: "σ / S cm⁻¹" → unit = "S cm⁻¹"
        - Example: "1000/T (K⁻¹)" → unit = "K⁻¹"
        
        Output valid JSON matching the FigureAnalysis schema."""

        response = await self._get_client().aio.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    parts=[
                        types.Part(text=prompt),
                        self._get_image_part(image_data)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self._sanitize_schema(FigureAnalysis.model_json_schema())
            )
        )
        
        try:
            result = FigureAnalysis.model_validate_json(response.text)
            result = result.model_dump()
            
            # Inject Usage Metadata
            if response.usage_metadata:
                result['_usage_metadata'] = {
                    'prompt_token_count': response.usage_metadata.prompt_token_count,
                    'candidates_token_count': response.usage_metadata.candidates_token_count,
                    'total_token_count': response.usage_metadata.total_token_count
                }
            
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return None
        
        if self.debug and self.save_debug:
            self._visualize_detection(image_path, result.get('subplots', []), query)

        return result

    def _visualize_detection(self, image_path: str, detections: List[Dict[str, Any]], query: str):
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        for i, box in enumerate(detections):
            left = (box['xmin'] / 1000) * w
            top = (box['ymin'] / 1000) * h
            right = (box['xmax'] / 1000) * w
            bottom = (box['ymax'] / 1000) * h
            
            color = "red" if i == 0 else "blue" # Alternate colors if helpful
            draw.rectangle([left, top, right, bottom], outline=color, width=5)
            
            label = box.get('label', f'Plot {i+1}')
            draw.text((left + 10, top + 10), label, fill=color)
        
        debug_path = self._get_debug_path(image_path, "_debug_detection", ext="png")
        img.save(debug_path)
        print(f"[DEBUG] Detection visualization saved to {debug_path}")

    def crop_image(self, image_path: str, box: Dict[str, Any], output_path: str = None, padding: int = 0, suffix: str = "_cropped") -> str:
        """
        Crops an image based on normalized coordinates (0-1000).
        Automatically adds safety padding if specified.
        """
        img = Image.open(image_path)
        width, height = img.size

        # Apply padding to normalized coordinates
        ymin = max(0, box['ymin'] - padding / 2)
        xmin = max(0, box['xmin'] - padding / 2)
        ymax = min(1000, box['ymax'] + padding)
        xmax = min(1000, box['xmax'] + padding)

        left = (xmin / 1000) * width
        top = (ymin / 1000) * height
        right = (xmax / 1000) * width
        bottom = (ymax / 1000) * height

        cropped_img = img.crop((left, top, right, bottom))
        
        # If output_path is not provided, generate a default one using the suffix
        should_save = False
        if output_path is None:
            output_path = self._get_debug_path(image_path, suffix)
            # Save if we are in debug mode OR if a specific suffix was requested (implies it's for extraction)
            if self.save_debug or suffix != "_cropped":
                should_save = True
        else:
            # If path is explicitly provided, we MUST save it (as it's usually needed for extraction)
            should_save = True
            
        if should_save:
            cropped_img.save(output_path)
            if self.debug:
                print(f"[DEBUG] Cropped image saved to {output_path}")
        elif self.debug:
            print(f"[DEBUG] Cropped image generated (but not saved) for {output_path}")
            
        return output_path

    def _slice_image(self, image_path: str, rows: int, cols: int) -> List[Dict[str, Any]]:
        """
        Slices an image into a grid of patches for grounding.
        """
        img = Image.open(image_path)
        width, height = img.size
        patch_width = width // cols
        patch_height = height // rows
        
        patches = []
        for r in range(rows):
            for c in range(cols):
                left = c * patch_width
                top = r * patch_height
                right = (c + 1) * patch_width
                bottom = (r + 1) * patch_height
                
                patch = img.crop((left, top, right, bottom))
                buffer = BytesIO()
                patch.save(buffer, format="JPEG")
                patch_data = buffer.getvalue()
                
                patches.append({
                    "data": patch_data,
                    "label": f"Grid [Row {r}, Col {c}]"
                })
        return patches

    def extract_data(self, image_path: str, grid_config: Optional[Dict[str, Any]] = None, prompt: Optional[str] = None, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts structured data from a scientific figure.
        Supports grid grounding by slicing the image into patches.
        """
        if prompt is None:
            ## Advanced Prompt
            prompt = f"""You are a high-precision scientific digitizer. Your job is to extract raw data from this plot with 100% completeness.
            
            {"CONTEXT FROM CAPTION: " + context if context else ""}

            CRITICAL INSTRUCTIONS:
            1. EXHAUSTIVE EXTRACTION: Do not summarize or sample. Extract EVERY single data marker visible on the plot, even if they overlap or are very close.
            2. LEGEND MAPPING: Read the figure legend. Do NOT use generic labels like "Series 1" or "Square". Map each data series to the EXACT text found in the legend (e.g., "Fe substitution").
            3. IONIC CONDUCTIVITY & LOG SCALES: 
               - Check the Y-axis values. If they are negative (e.g., -3, -4) but the unit labels look linear (e.g. "S/cm"), this is likely Log(Sigma).
               - If this matches, explicitly EXTRACT the unit as 'log(S/cm)' (or similar) instead of just 'S/cm'.
            4. DUAL Y-AXES:
               - Some plots have two Y-axes (left and right). Identify if this is the case.
               - Map data points to the correct Y-axis (e.g. circles to the left axis, squares to the right axis).
            5. X-AXIS IDENTIFICATION:
               - Check if the X-axis is Temperature (T, 1000/T) or Stoichiometry/Composition (x, y, z).
               - If it is stoichiometry (often labeled "x" or "z"), set 'xAxisType' to 'stoichiometry'.
               - If it is temperature (1000/T, Celsius, Kelvin), set 'xAxisType' to 'temperature'.
            6. TEMPERATURE AXIS specifics:
               - If xAxisType is 'temperature', ensure the extracted 'raw_temperature_unit' explicitly includes the format (e.g., "1000/T (K-1)").
            7. STOICHIOMETRY AXIS specifics:
               - If xAxisType is 'stoichiometry', extract the numeric value for 'xValue'. The 'raw_composition' for each series should ideally also incorporate this (e.g. "Al, x=0.2").
            8. INTERMEDIATE VALUES: Look specifically for data points falling BETWEEN major axis ticks.
            9. GRID USAGE: Use the provided high-resolution image slices to resolve dense clusters of points.
            
            Return the result in structured JSON format matching the schema."""
            ## Basic Prompt
            # prompt = """Analyze this scientific figure in high detail. 
            # 1. Identify the axes, their units, and the scale type (linear or log).
            # 2. Extract numerical data points from the chart/plot.
            # 3. Verify if this plot represents ionic conductivity measurements. 
            #    - Especially check if the X-axis is temperature or inverse temperature (e.g., 1000/T).
            #    - If temperature is not explicitly specified, assume the measurement was taken at room temperature (25 degrees Celsius) if it's a conductivity measurement.
            # 4. If multiple series exist, distinguish them using labels.
            # 5. If grid patches are provided, use them to verify tick marks and small text details.
            # Return the result in structured JSON format."""

        with open(image_path, "rb") as f:
            original_image_data = f.read()

        parts = [types.Part(text=prompt), self._get_image_part(original_image_data)]

        patches = []
        if grid_config and grid_config.get("enabled"):
            rows = grid_config.get("rows", 2)
            cols = grid_config.get("cols", 2)
            patches = self._slice_image(image_path, rows, cols)
            for patch in patches:
                parts.append(types.Part(text=f"Sub-image section: {patch['label']}"))
                parts.append(self._get_image_part(patch['data']))
        
        if self.debug and self.save_debug:
            # save the images in scifig_debug
            if grid_config and grid_config.get("enabled"):
                for patch in patches:
                    patch_name = patch['label'].replace(" ", "_").replace("[", "").replace("]", "").replace(",", "")
                    save_path = self._get_debug_path(image_path, f"_debug_extraction_input_{patch_name}")
                    with open(save_path, "wb") as f:
                        f.write(patch['data'])
            else:
                save_path = self._get_debug_path(image_path, "_debug_extraction_input")
                with open(save_path, "wb") as f:
                    f.write(original_image_data)

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING", "description": "Title of the scientific figure or chart."},
                "xAxis": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING", "description": "Label of the X axis."},
                        "unit": {"type": "STRING", "description": "Unit of measurement for X axis if available."},
                        "scale": {"type": "STRING", "enum": ["linear", "log"], "description": "Scale type of the X axis."},
                        "axisType": {"type": "STRING", "enum": ["temperature", "stoichiometry", "other"], "description": "The physical meaning of the X-axis."}
                    },
                    "required": ["label", "scale", "axisType"]
                },
                "yAxes": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "label": {"type": "STRING", "description": "Label of the Y axis."},
                            "unit": {"type": "STRING", "description": "Unit of measurement for Y axis if available."},
                            "scale": {"type": "STRING", "enum": ["linear", "log"], "description": "Scale type of the Y axis."}
                        },
                        "required": ["label", "scale"]
                    }
                },
                "dataPoints": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "label": {"type": "STRING", "description": "Legend label or category for this point."},
                            "xValue": {"type": "NUMBER", "description": "Numerical value on X axis."},
                            "yValue": {"type": "NUMBER", "description": "Numerical value on Y axis."},
                            "yAxisIndex": {"type": "INTEGER", "description": "The index (0-based) of the Y-axis this point belongs to."}
                        },
                        "required": ["label", "xValue", "yValue", "yAxisIndex"]
                    }
                },
                "summary": {"type": "STRING", "description": "Brief description of the findings in the figure."}
            },
            "required": ["title", "xAxis", "yAxes", "dataPoints", "summary"]
        }

        response = self._get_client().models.generate_content(
            model=self.model_name,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )

        result = json.loads(response.text)
        
        # Inject Usage Metadata
        if response.usage_metadata:
            result['_usage_metadata'] = {
                'prompt_token_count': response.usage_metadata.prompt_token_count,
                'candidates_token_count': response.usage_metadata.candidates_token_count,
                'total_token_count': response.usage_metadata.total_token_count
            }
        
        if self.debug and self.save_debug:
            self._visualize_extraction(image_path, result)
            
        return result

    async def extract_data_async(self, 
        image_path: str, 
        grid_config: Optional[Dict[str, Any]] = None, 
        context: Optional[str] = None,
        axis_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts structured data from a scientific figure (Asynchronous).
        """
        # We explicitly tell the model what "Left" and "Right" represent.
        x_def = "Unknown"
        left_def = "Unknown" 
        right_def = "None"
        
        if axis_hints:
            x_axis = axis_hints.get('x_axis', {})
            x_quantity = x_axis.get('quantity_type', 'other')
            x_range = x_axis.get('value_range')
            x_def = f"{x_axis.get('title_text', 'X-Axis')} (Type: {x_quantity}, Unit: {x_axis.get('unit', 'N/A')})"
            if x_range:
                x_def += f" [Range: {x_range.get('min', '?')} to {x_range.get('max', '?')}]"
            
            # Left Y-Axis details
            l_axis = axis_hints.get('left_y_axis', {})
            l_range = l_axis.get('value_range')
            left_def = f"{l_axis.get('title_text', 'Y-Axis')} (Type: {l_axis.get('quantity_type', 'other')}, Unit: {l_axis.get('unit', 'N/A')})"
            if l_range:
                left_def += f" [Range: {l_range.get('min', '?')} to {l_range.get('max', '?')}]"
            
            # Right Y-Axis details (if it exists)
            r_axis = axis_hints.get('right_y_axis')
            if r_axis:
                r_range = r_axis.get('value_range')
                right_def = f"{r_axis.get('title_text', 'Secondary Y')} (Unit: {r_axis.get('unit', 'N/A')})"
                if r_range:
                    right_def += f" [Range: {r_range.get('min', '?')} to {r_range.get('max', '?')}]"

        prompt = f"""You are a data extraction engine. Extract numerical measurements/points from this plot.

            The axes have already been identified. You must map points to them:
            
            - X-AXIS: {x_def}
            - LEFT Y-AXIS: {left_def}
            - RIGHT Y-AXIS: {right_def}
            
            {"CONTEXT FROM CAPTION: " + context if context else ""}
            1. SERIES IDENTIFICATION:
       - Use EXACT text from the legend/key
       - DO NOT use generic labels like "Series 1" or "Data 1"
       - Examples: "Fe substitution", "x=0.2", "Cold-pressed", "223K"
    
    2. LEGEND INTERPRETATION (CONTEXT-DEPENDENT):
       - In FREQUENCY plots (X=Hz): Legend shows temperatures ("223K", "263K")
         → These are experimental conditions, NOT material composition
         → DO NOT extract data from frequency plots
       - In ARRHENIUS plots (X=1000/T): Legend shows materials ("Fe doped", "Al doped")
         → These are material variants to extract
    
    3. DATA EXTRACTION:
       - Extract EVERY visible data point (do not sample)
       - Create parallel lists: x_values and y_values (same length)
       - Preserve ALL intermediate points between tick marks
    
    4. Y-AXIS ROUTING:
       - If a series uses the LEFT Y-axis unit → mapped_y_axis="left"
       - If a series uses the RIGHT Y-axis unit → mapped_y_axis="right"
       - If only one Y-axis exists → always use "left"
    
    5. COMPLETENESS:
       - If plot has 50 data points, extract all 50
       - Do not summarize or skip dense regions

    6. TEMPERATURE ANNOTATION:
       - If NEITHER axis represents temperature (e.g., this is a composition vs conductivity plot),
         look for a temperature annotation INSIDE the plot area, title, subtitle, or corner text
         (e.g., "at 70°C", "T = 300K", "room temperature", "(RT)")
       - If found, set annotated_temperature to the value (e.g., "70°C", "300K", "25°C")
       - If not found, set annotated_temperature to null
       - If an axis already represents temperature, set annotated_temperature to null

    Return structured JSON matching the ExtractionResult schema."""

        with open(image_path, "rb") as f:
            original_image_data = f.read()

        parts = [types.Part(text=prompt), self._get_image_part(original_image_data)]

        patches = []
        if grid_config and grid_config.get("enabled"):
            rows = grid_config.get("rows", 2)
            cols = grid_config.get("cols", 2)
            patches = self._slice_image(image_path, rows, cols)
            for patch in patches:
                parts.append(types.Part(text=f"Sub-image section: {patch['label']}"))
                parts.append(self._get_image_part(patch['data']))
        
        if self.debug and self.save_debug:
            if grid_config and grid_config.get("enabled"):
                for patch in patches:
                    patch_name = patch['label'].replace(" ", "_").replace("[", "").replace("]", "").replace(",", "")
                    save_path = self._get_debug_path(image_path, f"_debug_extraction_input_{patch_name}")
                    with open(save_path, "wb") as f:
                        f.write(patch['data'])
            else:
                save_path = self._get_debug_path(image_path, "_debug_extraction_input")
                with open(save_path, "wb") as f:
                    f.write(original_image_data)

        response = await self._get_client().aio.models.generate_content(
            model=self.model_name,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self._sanitize_schema(ExtractionResult.model_json_schema())
            )
        )

        try:
            result_obj = ExtractionResult.model_validate_json(response.text)
            result = result_obj.model_dump()

            # Validate before returning
            result = self._validate_extraction(result, axis_hints)

            if axis_hints:
                result['axis_metadata'] = {
                    'x': axis_hints.get('x_axis'),
                    'left': axis_hints.get('left_y_axis'),
                    'right': axis_hints.get('right_y_axis')
                }
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            result = {}
        
        # Inject Usage Metadata (even if parsing failed, we paid for tokens)
        if response.usage_metadata:
            result['_usage_metadata'] = {
                'prompt_token_count': response.usage_metadata.prompt_token_count,
                'candidates_token_count': response.usage_metadata.candidates_token_count,
                'total_token_count': response.usage_metadata.total_token_count
            }

        if self.debug and self.save_debug:
            self._visualize_extraction(image_path, result)
            
        return result

    def _validate_extraction(self, result: Dict[str, Any], axis_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate extraction results against axis hints, including value range bounds."""
        
        if not axis_hints:
            return result
        
        x_quantity = axis_hints.get('x_axis', {}).get('quantity_type', 'other')
        
        # Filter out invalid extractions
        if x_quantity == 'frequency':
            result['warnings'] = result.get('warnings', [])
            result['warnings'].append(
                "CRITICAL: X-axis is frequency. This data should NOT have been extracted. "
                "Please report this as a detection error."
            )
            result['data_series'] = []  # Clear all data
        
        # Validate series labels for temperature plots
        if x_quantity in ['temperature_inverse', 'temperature_absolute']:
            for series in result.get('data_series', []):
                label = series.get('series_label', '')
                # Check if label looks like temperature (e.g., "223K", "300K")
                if re.match(r'^\d+K$', label):
                    if 'warnings' not in result:
                        result['warnings'] = []
                    result['warnings'].append(
                        f"Series '{label}' looks like temperature but X-axis is also temperature. "
                        f"This might indicate wrong subplot was extracted."
                    )
        
        # --- NEW: Axis-bounds validation ---
        # Check if extracted values fall within the detected axis range
        x_range = axis_hints.get('x_axis', {}).get('value_range')
        y_range = axis_hints.get('left_y_axis', {}).get('value_range')
        
        if x_range or y_range:
            for series in result.get('data_series', []):
                x_vals = series.get('x_values', [])
                y_vals = series.get('y_values', [])
                
                filtered_x = []
                filtered_y = []
                oob_count = 0
                
                for x, y in zip(x_vals, y_vals):
                    x_ok = True
                    y_ok = True
                    
                    if x_range:
                        x_min = x_range.get('min')
                        x_max = x_range.get('max')
                        if x_min is not None and x_max is not None:
                            # Allow 15% margin for edge points
                            margin = abs(x_max - x_min) * 0.15
                            if x < x_min - margin or x > x_max + margin:
                                x_ok = False
                    
                    if y_range:
                        y_min = y_range.get('min')
                        y_max = y_range.get('max')
                        if y_min is not None and y_max is not None:
                            margin = abs(y_max - y_min) * 0.15
                            if y < y_min - margin or y > y_max + margin:
                                y_ok = False
                    
                    if x_ok and y_ok:
                        filtered_x.append(x)
                        filtered_y.append(y)
                    else:
                        oob_count += 1
                
                if oob_count > 0:
                    if 'warnings' not in result:
                        result['warnings'] = []
                    result['warnings'].append(
                        f"Removed {oob_count} out-of-bounds points from series '{series.get('series_label', '?')}'"
                    )
                    series['x_values'] = filtered_x
                    series['y_values'] = filtered_y
        
        return result
        
    def _visualize_extraction(self, image_path: str, result: Dict[str, Any]):
        """
        Plots the extracted data side-by-side with the original input figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot original image
        img = Image.open(image_path)
        ax1.imshow(img)
        ax1.set_title("Input Figure to Extraction")
        ax1.axis('off')
        
        # Plot extracted data
        data_points = result.get('dataPoints', [])
        y_axes = result.get('yAxes', [])
        labels = list(set(dp['label'] for dp in data_points))
        
        # Create secondary axis if needed
        is_dual = len(y_axes) > 1
        ax2_sec = None
        if is_dual:
            ax2_sec = ax2.twinx()

        for label in labels:
            # Group by yAxisIndex
            for y_idx in range(len(y_axes)):
                x = [dp['xValue'] for dp in data_points if dp['label'] == label and dp.get('yAxisIndex', 0) == y_idx]
                y = [dp['yValue'] for dp in data_points if dp['label'] == label and dp.get('yAxisIndex', 0) == y_idx]
                
                if x:
                    target_ax = ax2_sec if y_idx == 1 and ax2_sec else ax2
                    marker = 'o' if y_idx == 0 else 's'
                    target_ax.scatter(x, y, label=f"{label} (Axis {y_idx})", marker=marker)
            
        x_label = result.get('xAxis', {}).get('label', 'X')
        x_unit = result.get('xAxis', {}).get('unit', '')
        x_scale = result.get('xAxis', {}).get('scale', 'linear')
        
        ax2.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
        if x_scale == 'log':
            ax2.set_xscale('log')

        for i, y_ax in enumerate(y_axes):
            target_ax = ax2_sec if i == 1 and ax2_sec else ax2
            y_label = y_ax.get('label', 'Y')
            y_unit = y_ax.get('unit', '')
            y_scale = y_ax.get('scale', 'linear')
            
            target_ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)
            if y_scale == 'log':
                target_ax.set_yscale('log')
            
        ax2.set_title(result.get('title', 'Extracted Data'))
        
        # Merge legends if dual
        if is_dual:
            lines, labels = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_sec.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2)
        elif labels:
            ax2.legend()
            
        ax2.grid(True, linestyle='--', alpha=0.7, which="both")
        
        debug_path = self._get_debug_path(image_path, "_debug_extraction", ext="png")
        plt.tight_layout()
        plt.savefig(debug_path)
        plt.close()
        print(f"[DEBUG] Extraction visualization saved to {debug_path}")

if __name__ == "__main__":
    # Example usage (requires API_KEY in env)
    import sys
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        sys.exit(1)
        
    parser = SciFigureParser(api_key=api_key, debug=True)
    # figure_path = "path/to/your/figure.jpg"
    # box = parser.detect_subplot(figure_path, "ionic conductivity")
    # cropped_path = parser.crop_image(figure_path, box, padding=80)
    # result = parser.extract_data(cropped_path, grid_config={"enabled": True, "rows": 2, "cols": 2})
    # print(json.dumps(result, indent=2))
