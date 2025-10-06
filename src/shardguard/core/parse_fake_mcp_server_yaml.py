""" 
   This library will read in the needed information to fake MCP servers from a 
   series of YAML files.  The YAML files must be in the format:

   server: <server_name>
   description: <description>
   tools:
     - name: <tool_name>            <- can be repeated multiple times
       description: <tool_description>
       properties:
         - name: <property_name>    <- can be repeated multiple times
           type: <property_type>
           description: <property_description>
"""

import yaml

ALLOWED_TYPES = [
    "string",
    "number",
    "boolean",
    "null",
    "array",
    "integer",
    "object"   # Have updaed this for the reason that web server has a parameter that gets object of headers
]



def parse_fake_MCP_YAML_directory(directory):
    """
    Parses all YAML files in the given directory to create a fake MCP server configuration.

    This function reads each YAML file, validates its structure, and extracts the server,
    description, and tools information. It returns a list of dictionaries representing
    the parsed server configurations.
    """
    import os

    if not os.path.isdir(directory):
        raise ValueError(f"Directory '{directory}' does not exist or is not a directory.")

    server_configs = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            full_path = os.path.join(directory, filename)
            try:
                config = parse_fake_MCP_YAML_file(full_path)
                server_configs.append(config)
            except ValueError as e:
                print(f"Error parsing file '{filename}': {e}")

    # I need to ensure that the server names are unique
    server_names = set()
    for config in server_configs:
        server_name = config["server"]
        if server_name in server_names:
            raise ValueError(f"Duplicate server name found: '{server_name}'")
        server_names.add(server_name)

    return server_configs



def parse_fake_MCP_YAML_file(filename):
    """
    Parses the YAML data for a fake MCP server configuration.

    This function has a lot of strict validation rules to ensure that the
    YAML data adheres to a specific structure and contains the required fields.
    """

    def check_exact_fields(obj, allowed_keys, context="root"):
        # helper that ensures that exactly these fields are present
        extra_keys = set(obj.keys()) - set(allowed_keys)
        if extra_keys:
            raise ValueError(f"{context}: unexpected field(s): {', '.join(extra_keys)}")
        
        missing_keys = set(allowed_keys) - set(obj.keys())
        if missing_keys:
            raise ValueError(f"{context}: missing required field(s): {', '.join(missing_keys)}")

    with open(filename, 'r') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file '{filename}': {e}")

    if not isinstance(data, dict):
        raise ValueError("YAML root must be a dictionary.")

    # Validate top-level fields
    required_top = ["server", "description", "tools"]
    check_exact_fields(data, required_top, context="Top-level")

    server = data["server"]
    description = data["description"]
    tools_raw = data["tools"]

    if not isinstance(tools_raw, list):
        raise ValueError("The 'tools' field must be a list.")

    tool_names = set()
    tools = []

    # iterate into each tool...
    for idx, tool in enumerate(tools_raw):
        context = f"Tool[{idx}]"
        if not isinstance(tool, dict):
            raise ValueError(f"{context}: each tool must be a dictionary.")

        check_exact_fields(tool, ["name", "description", "properties"], context)

        name = tool.get("name")
        if name is None:
            raise ValueError(f"{context}: missing 'name'")
        if name in tool_names:
            raise ValueError(f"{context}: duplicate tool name: {name}")
        tool_names.add(name)

        tool_description = tool.get("description")
        if tool_description is None:
            raise ValueError(f"{context}: missing 'description' for tool '{name}'")
        if not isinstance(tool_description, str):
            raise ValueError(f"{context}: 'description' for tool '{name}' must be a string")

        properties_raw = tool.get("properties", [])
        if not isinstance(properties_raw, list):
            raise ValueError(f"{context}: 'properties' must be a list")

        property_names = set()
        properties = []

        # iterate into the properties for each tool...
        for p_idx, prop in enumerate(properties_raw):
            p_context = f"{context} -> Property[{p_idx}]"
            if not isinstance(prop, dict):
                raise ValueError(f"{p_context}: must be a dictionary.")
            check_exact_fields(prop, ["name", "type", "description"], p_context)

            prop_name = prop.get("name")
            prop_type = prop.get("type")
            prop_desc = prop.get("description")

            if prop_name is None or prop_type is None or prop_desc is None:
                raise ValueError(f"{p_context}: missing required field(s)")

            if prop_type not in ALLOWED_TYPES:
                raise ValueError(f"{p_context}: invalid type '{prop_type}'")

            if prop_name in property_names:
                raise ValueError(f"{p_context}: duplicate property name: '{prop_name}'")
            property_names.add(prop_name)

            properties.append({
                "name": prop_name,
                "type": prop_type,
                "description": prop_desc,
            })

        tools.append({
            "name": name,
            "description": tool_description,
            "properties": properties
        })

    return {
        "server": server,
        "description": description,
        "tools": tools
    }


