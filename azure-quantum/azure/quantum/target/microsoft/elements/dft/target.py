import warnings

from azure.quantum.job.base_job import ContentType
from azure.quantum.job.job import Job
from azure.quantum.target.target import Target
from azure.quantum.workspace import Workspace
from azure.quantum.target.params import InputParams
from typing import Any, Dict, Type, Union, List
from .job import MicrosoftElementsDftJob
from pathlib import Path
import copy
import json
from collections import defaultdict


class MicrosoftElementsDft(Target):
    """
    Microsoft Elements Dft target from the microsoft-elements provider.
    """

    target_names = [
        "microsoft.dft"
    ]


    def __init__(
        self,
        workspace: "Workspace",
        name: str = "microsoft.dft",
        **kwargs
    ):
        """
        Initializes a new DFT target.

        :param workspace: Associated workspace
        :type workspace: Workspace
        :param name: Target name
        """
        # There is only a single target name for this target
        assert name == self.target_names[0]

        # make sure to not pass argument twice
        kwargs.pop("provider_id", None)

        super().__init__(
            workspace=workspace,
            name=name,
            input_data_format="microsoft.xyz.v1",
            output_data_format="microsoft.dft-results.v1",
            provider_id="microsoft-elements",
            content_type=ContentType.text_plain,
            **kwargs
        )


    def submit(self,
               input_data: Any,
               name: str = "azure-quantum-dft-job",
               shots: int = None,
               input_params: Union[Dict[str, Any], InputParams, None] = None,
               **kwargs) -> MicrosoftElementsDftJob:
        """
        Submit DFT job to Azure Quantum Services.
        
        :param input_data: Input data. Can be:
                           - An XYZ string
                           - A file path to an XYZ or QCSchema file
                           - A list of file paths to XYZ or QCSchema files
                           - A list of QCSchema dictionaries
                           - A list of XYZ strings
        :type input_data: Any
        :param name: Job name
        :type name: str
        :param shots: Number of shots. Ignored in DFT job. Defaults to None
        :type shots: int
        :param input_params: Input parameters
        :type input_params: Dict[str, Any]
        :return: Azure Quantum job
        :rtype: Job
        """

        if shots is not None:
            warnings.warn("The 'shots' parameter is ignored in Microsoft Elements Dft job.")
            
        # Helper function to detect if a string is likely XYZ content
        def is_xyz_content(s):
            try:
                # XYZ files shouldn't look like JSON
                if '{' in s or '}' in s:
                    return False
                    
                lines = s.strip().split('\n')
                if len(lines) < 3:
                    return False
                    
                # Try to parse first line as an integer (number of atoms)
                num_atoms = int(lines[0].strip())
                
                # Check if we have at least as many atom lines as expected
                content_lines = [line for line in lines[2:] if line.strip()]
                if len(content_lines) < num_atoms:
                    return False
                
                return True
            except (ValueError, AttributeError):
                return False
        
        if isinstance(input_data, list):
            # Handle list input (files or QCSchema objects)
            if len(input_data) < 1:
                raise ValueError("Input data list has no elements.")

            if all(isinstance(task, str) for task in input_data):
                # First check if all items look like XYZ content
                all_xyz_content = all(is_xyz_content(item) for item in input_data)
                
                # Or check if any items are XYZ content but don't exist as files
                any_xyz_not_file = any(is_xyz_content(item) and not Path(item).exists() for item in input_data)
                
                if all_xyz_content or any_xyz_not_file:
                    # List contains XYZ strings, not file paths
                    # Convert each XYZ string to QCSchema
                    qcschema_data = []
                    for i, xyz_content in enumerate(input_data):
                        try:
                            mol = self._xyz_to_qcschema_mol(xyz_content)
                            qcschema = self._new_qcshema(input_params or {}, mol)
                            qcschema_data.append(qcschema)
                        except ValueError as e:
                            # Provide more detailed error for specific item
                            raise ValueError(f"Error processing XYZ content at index {i}: {str(e)}. Please check the format of your XYZ data.") from e
                    
                    # Create blobs for each QCSchema
                    qcschema_blobs = {}
                    for i in range(len(qcschema_data)):
                        qcschema_blobs[f"inputData_{i}"] = self._encode_input_data(qcschema_data[i])
                    
                    # Use placeholder file names for table of contents
                    placeholder_names = [f"content_{i}.xyz" for i in range(len(input_data))]
                    toc_str = self._create_table_of_contents(placeholder_names, list(qcschema_blobs.keys()))
                else:
                    # List of file paths
                    try:
                        qcschema_data = self.assemble_qcschema_from_files(input_data, input_params)
                    except FileNotFoundError as e:
                        raise ValueError(f"File not found error: {str(e)}. Please ensure all file paths are correct.") from e
                    except ValueError as e:
                        raise ValueError(f"Error processing input files: {str(e)}. Please check your file formats and ensure they are valid XYZ or QCSchema files.") from e
                
                    qcschema_blobs = {}
                    for i in range(len(qcschema_data)):
                        qcschema_blobs[f"inputData_{i}"] = self._encode_input_data(qcschema_data[i])
                
                    toc_str = self._create_table_of_contents(input_data, list(qcschema_blobs.keys()))
            elif all(isinstance(task, dict) for task in input_data): 
                # List of QCSchema dictionaries
                qcschema_blobs = {}
                try:
                    for i in range(len(input_data)):
                        qcschema_blobs[f"inputData_{i}"] = self._encode_input_data(input_data[i])
                    toc_str = '{"description": "QcSchema Objects were given for input."}'
                except Exception as e:
                    raise ValueError(f"Error processing QCSchema dictionary at index {i}: {str(e)}. Please ensure all dictionaries are valid QCSchema objects.") from e
            else:
                # Check if we have a mix of strings and dictionaries
                if any(isinstance(task, str) for task in input_data) and any(isinstance(task, dict) for task in input_data):
                    raise ValueError("Mixed string and dictionary types in input_data list. Please use either all strings (file paths or XYZ content) or all dictionaries (QCSchema objects).")
                # Get the actual types for better error reporting
                types = [type(item).__name__ for item in input_data[:3]]
                if len(input_data) > 3:
                    types.append("...")
                raise ValueError(f"Unsupported batch submission with types: {', '.join(types)}. Please use either List[str] for file paths/XYZ content or List[dict] for QCSchema objects.")
                
            toc = self._encode_input_data(toc_str)

            input_params = {} if input_params is None else input_params
            return self._get_job_class().from_input_data_container(
                workspace=self.workspace,
                name=name,
                target=self.name,
                input_data=toc,
                batch_input_blobs=qcschema_blobs,
                input_params={ 'numberOfFiles': len(input_data), "inputFiles": list(qcschema_blobs.keys()), **input_params },
                content_type=kwargs.pop('content_type', self.content_type),
                encoding=kwargs.pop('encoding', self.encoding),
                provider_id=self.provider_id,
                input_data_format=kwargs.pop('input_data_format', 'microsoft.qc-schema.v1'),
                output_data_format=kwargs.pop('output_data_format', self.output_data_format),
                session_id=self.get_latest_session_id(),
                **kwargs
            )
        elif isinstance(input_data, str):
            # Handle string input (file path or XYZ content)
            file_path = Path(input_data)
            
            # Check if this is XYZ content or a file path
            if is_xyz_content(input_data) and not file_path.exists():
                # It's XYZ content - convert to QCSchema
                try:
                    mol = self._xyz_to_qcschema_mol(input_data)
                    qcschema = self._new_qcshema(input_params or {}, mol)
                    
                    # Submit as JSON QCSchema
                    return super().submit(
                        input_data=qcschema,
                        name=name, 
                        shots=shots, 
                        input_params=input_params,
                        content_type=ContentType.json,
                        input_data_format='microsoft.qc-schema.v1',
                        **kwargs
                    )
                except ValueError as e:
                    raise ValueError(f"Error processing XYZ content: {str(e)}. Please check your XYZ data format.") from e
            elif file_path.exists():
                # It's an existing file path
                if file_path.suffix.lower() == '.xyz':
                    try:
                        # For XYZ files, convert to QCSchema first
                        file_data = file_path.read_text()
                        mol = self._xyz_to_qcschema_mol(file_data)
                        qcschema = self._new_qcshema(input_params or {}, mol)
                        
                        return super().submit(
                            input_data=qcschema,
                            name=name, 
                            shots=shots, 
                            input_params=input_params,
                            content_type=ContentType.json,
                            input_data_format='microsoft.qc-schema.v1',
                            **kwargs
                        )
                    except ValueError as e:
                        raise ValueError(f"Error processing XYZ file '{file_path}': {str(e)}. Please check your file format.") from e
                elif file_path.suffix.lower() == '.json':
                    try:
                        # For JSON files, assume they're already in QCSchema format
                        with open(file_path, 'r') as f:
                            qcschema = json.load(f)
                        
                        return super().submit(
                            input_data=qcschema,
                            name=name, 
                            shots=shots, 
                            input_params=input_params,
                            content_type=ContentType.json,
                            input_data_format='microsoft.qc-schema.v1',
                            **kwargs
                        )
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Error decoding JSON file '{file_path}': {str(e)}. Please ensure it's a valid JSON file.") from e
                    except Exception as e:
                        raise ValueError(f"Error processing file '{file_path}': {str(e)}") from e
                else:
                    # Unknown file extension
                    raise ValueError(f"Unsupported file type: {file_path.suffix}. Please use .xyz or .json files.")
            else:
                # Not recognized as XYZ content and not an existing file
                raise ValueError(f"Input string '{input_data[:40]}...' (truncated) is neither recognized as XYZ content nor exists as a file. Please provide valid XYZ content or a correct file path.")
        elif isinstance(input_data, dict):
            # Handle dictionary input - check if it's a QCSchema
            if 'schema_name' in input_data and input_data['schema_name'].startswith(('qcschema_', 'madft_')):
                # It's a QCSchema dictionary - submit it directly with proper content type
                return super().submit(
                    input_data=input_data,
                    name=name, 
                    shots=shots,
                    input_params=input_params,
                    content_type=ContentType.json,
                    input_data_format='microsoft.qc-schema.v1',
                    **kwargs
                )
            else:
                raise ValueError(f"Invalid dictionary input: Dictionary does not appear to be a QCSchema (missing 'schema_name' field or incorrect schema type). For DFT jobs, input_data must be a file path, XYZ content, or QCSchema.")
        else:
            raise ValueError(f"Invalid input_data type: {type(input_data).__name__}. Expected a file path, XYZ content string, a list of these, or a QCSchema dictionary.")


    
    @classmethod
    def assemble_qcschema_from_files(self, input_data: Union[List[str]], input_params: Dict) -> List[Dict]:
        """
        Convert a list of files to a list of QcSchema objects that are ready for submission.
        
        :param input_data: Input data
        :type input_data: List[str]
        :param input_params: Input parameters
        :type input_params: Dict[str, Any]
        :rtype: List[Dict]
        """

        self._check_file_paths(input_data)

        qcshema_objects = []
        for file in input_data:
            file_path = Path(file)
            
            file_data = file_path.read_text()
            if file_path.suffix == '.xyz':
                mol = self._xyz_to_qcschema_mol(file_data)
                new_qcschema = self._new_qcshema( input_params, mol )
                qcshema_objects.append(new_qcschema)
            elif file_path.suffix == '.json':
                if input_params is not None and len(input_params.keys()) > 0:
                    warnings.warn('Input parameters were given along with a QcSchema file which contains parameters, using QcSchema parameters as is.')
                with open(file_path, 'r') as f:
                    qcshema_objects.append( json.load(f) )
            else:
                raise ValueError(f"File type '{file_path.suffix}' for file '{file_path}' is not supported.")

        return qcshema_objects

    @classmethod
    def _check_file_paths( self, input_data: List[str]):
        """Check the file types and make sure they are supported by our parsers."""

        warn_task_count = 1000
        if len(input_data) >= warn_task_count:
            warnings.warn(f'Number of tasks is greater than {warn_task_count}.')

        supported_ext = ['.xyz', '.json']
        prev_ext = None
        for path_str in input_data:
            path = Path(path_str)

            if not path.exists():
                raise FileNotFoundError(f"File {path_str} does not exist.")

            if path.suffix not in supported_ext:
                raise ValueError(f"'{path.suffix}' file type is not supported. Please use one of {supported_ext}.")
            
            if prev_ext is not None and prev_ext !=  path.suffix:
                raise ValueError(f"Multiple file types were provided ('{path.suffix}', '{prev_ext}'). Please submit only one file type.")
            else:
                prev_ext = path.suffix
            

    @classmethod
    def _new_qcshema( self, input_params: Dict[str,Any], mol: Dict[str,Any],  ) -> Dict[str, Any]:
        """
        Create a new default qcshema object.
        """

        self._sanity_check_params(input_params, mol)

        if input_params.get("driver").lower() == "go":
            copy_input_params = copy.deepcopy(input_params)
            copy_input_params["driver"] = "gradient"
            new_object = {
                "schema_name": "qcschema_optimization_input",
                "schema_version": 1,
                "initial_molecule": mol,
            }
            if copy_input_params.get("go_keywords"):
                new_object["keywords"] = copy_input_params.pop("go_keywords")
            new_object["input_specification"] = copy_input_params
            return new_object
        elif input_params.get("driver").lower() == "bomd":
            copy_input_params = copy.deepcopy(input_params)
            copy_input_params["driver"] = "gradient"
            new_object = {
                "schema_name": "madft_molecular_dynamics_input",
                "schema_version": 1,
                "initial_molecule": mol,
            }
            if copy_input_params.get("bomd_keywords"):
                new_object["keywords"] = copy_input_params.pop("bomd_keywords")
            new_object["input_specification"] = copy_input_params
            return new_object
        else:
            new_object = copy.deepcopy(input_params)
            new_object.update({
                "schema_name": "qcschema_input",
                "schema_version": 1,
                "molecule": mol,
            })
            return new_object
    
    @classmethod
    def _sanity_check_params(self, input_params, mol):

        # QM/MM is not supported for GO, BOMD and Hessian.
        driver = input_params.get("driver",'').lower()
        if driver in ["go", "bomd", "hessian"]:
            if "extras" in mol and "mm_charges" in mol["extras"]:
                raise ValueError(f"'{driver}' does not support QM/MM.")

        # Top level params
        self._check_dict_for_required_keys(input_params, 'input_params', ['driver', 'model'])
        
        # Check Model params
        self._check_dict_for_required_keys(input_params['model'], 'input_params["model"]', ['method', 'basis'])

        supported_drivers = ['energy', 'gradient', 'hessian', 'go', 'bomd']
        if input_params['driver'] not in supported_drivers:
            raise ValueError(f"Driver ({input_params['driver']}) is not supported. Please use one of {supported_drivers}.")

    
    @classmethod
    def _check_dict_for_required_keys(self, input_params: dict, dict_name: str, required_keys: list[str]):
        """Check dictionary for required keys and if it doesn't have then raise ValueError."""

        for required_key in required_keys:
            if required_key not in input_params.keys():
                raise ValueError(f"Required key ({required_key}) was not provided in {dict_name}.")

    @classmethod
    def _xyz_to_qcschema_mol(self, file_data: str ) -> Dict[str, Any]:
        """
        Convert xyz format to qcschema molecule.
        """

        lines = file_data.split("\n")
        if len(lines) < 3 or not lines[0]:
            raise ValueError("Invalid xyz format.")
        n_atoms = int(lines.pop(0))
        comment = lines.pop(0)
        mol = defaultdict(list)
        mol['extras'] = defaultdict(list)
        bohr_to_angstrom = 0.52917721092
        for line in lines:
            if line:
                elements = line.split()
                if len(elements) == 4:
                    symbol, x, y, z = elements
                    mol["symbols"].append(symbol)
                    mol["geometry"] += [float(x)/bohr_to_angstrom, float(y)/bohr_to_angstrom, float(z)/bohr_to_angstrom]
                elif len(elements) == 5:
                    symbol, x, y, z, q = elements
                    if symbol[0] != '-':
                        raise ValueError("Invalid xyz format. Molecular Mechanics atoms requires '-' at the beginning of the atom type.")
                    mol["extras"]["mm_symbols"].append(symbol.replace('-', ''))
                    mol["extras"]["mm_geometry"] += [float(x)/bohr_to_angstrom, float(y)/bohr_to_angstrom, float(z)/bohr_to_angstrom]
                    mol["extras"]["mm_charges"].append(float(q))
                else:
                    raise ValueError("Invalid xyz format.")
            else:
                break

        # Convert defaultdict to dict
        mol = dict(mol)
        mol["extras"] = dict(mol["extras"])
        
        if len(mol["symbols"])+len(mol["extras"].get("mm_symbols",[])) != n_atoms:
            raise ValueError("Number of inputs does not match the number of atoms in xyz file.")

        return mol

    @classmethod
    def _get_job_class(cls) -> Type[Job]:
        return MicrosoftElementsDftJob

    @classmethod
    def _create_table_of_contents(cls, input_files: List[str], input_blobs: List[str]) -> Dict[str,Any]:
        """Create the table of contents for a batched job that contains a description of file and the mapping between the file names and the blob names"""

        assert len(input_files) == len(input_blobs), "Internal error: number of blobs is not that same as the number of files."

        toc = []
        for i in range(len(input_files)):
            toc.append( 
                {
                    "inputFileName": input_files[i],
                    "qcschemaBlobName": input_blobs[i],
                }
            )

        return {
            "description": "This files contains the mapping between the xyz file name that were submitted and the qcschema blobs that are used for the calculation.",
            "tableOfContents": toc,
        }
