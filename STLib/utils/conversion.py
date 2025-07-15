import warnings

def naifSchemaToExtended(naifID: str):

    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html#Asteroids
    
    #                 |                Original                   |                Extended
    # Single-body <<<<|-------------------------------------------|-------------------------------------------
    #                 |   2000000 + Permanent Asteroid Number     |   20000000 + Permanent Asteroid Number
    #                 |   3000000 + Provisional Asteroid Number   |   50000000 + Provisional Asteroid Number
    # Multi-body <<<<<|-------------------------------------------|-------------------------------------------
    #   Barycenters   |                N/A                        |           optional prefix 0
    #   Primary Body  |                N/A                        |                prefix 9
    #   Satellites    |                N/A                        |                prefix 1-8

    if not naifID.isdigit():
        raise ValueError("Expected integer string NAIF ID.")

    # single body case
    if len(naifID) == 7:
        if naifID[0] == '2':
            return '20' + naifID[1:]
        if naifID[0] == '3':
            return '50' + naifID[1:]
    
    # multi-body barycenter, no prefix
    elif len(naifID) == 8:
        if naifID[0] not in "01":
            warnings.warn("The provided NAIF ID is already in the extended schema.")
            return naifID
        
    # multi-body with prefix
    elif len(naifID) == 9:
        if naifID[1] not in "01":
            warnings.warn("The provided NAIF ID is already in the extended schema.")
            return naifID

    raise ValueError("Expected NAIF ID string in either original or extended schema.")


def naifSchemaToOriginal(naifID):

    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html#Asteroids
    
    #                 |                Original                   |                Extended
    # Single-body <<<<|-------------------------------------------|-------------------------------------------
    #                 |   2000000 + Permanent Asteroid Number     |   20000000 + Permanent Asteroid Number
    #                 |   3000000 + Provisional Asteroid Number   |   50000000 + Provisional Asteroid Number
    # Multi-body <<<<<|-------------------------------------------|-------------------------------------------
    #   Barycenters   |                N/A                        |           optional prefix 0
    #   Primary Body  |                N/A                        |                prefix 9
    #   Satellites    |                N/A                        |                prefix 1-8
    
    if not naifID.isdigit():
        raise ValueError("Expected integer string NAIF ID.")
    
    if len(naifID) == 7 and naifID[0] in "23":
        warnings.warn("The provided NAIF ID is already in the original schema.")
        return naifID
    
    # single body case
    if len(naifID) == 8 and naifID[0] not in "01":
        if naifID[0:2] == '20':
            return '2' + naifID[2:]
        if naifID[0:2] == '50':
            return '3' + naifID[2:]
        # exceeded numbering capacity of original schema
        raise ValueError(f"The extended NAIF ID {naifID} cannot be accommodated by the original schema.")
    
    # multi-body case
    elif len(naifID) == 9:
        raise ValueError("Cannot convert a multi-body NAIF ID to original schema.")

    raise ValueError("Expected NAIF ID string in either original or extended schema.")