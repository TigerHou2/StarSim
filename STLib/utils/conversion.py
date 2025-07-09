def naifSchemaToExtended(naifID):
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html#Asteroids
    # expect string
    if len(naifID) != 7:
        if len(naifID) == 8:
            # assume we are already in extended schema
            return naifID
        raise ValueError('Expected original schema with 7 digits.')
    if naifID[0] == '2':
        return '20' + naifID[1:]
    if naifID[0] == '3':
        return '50' + naifID[1:]
    raise ValueError('Expected 2 or 3 as leading digit in the original schema.')


def naifSchemaToOriginal(naifID):
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html#Asteroids
    # expect string
    if len(naifID) != 8:
        if len(naifID) == 7:
            # assume we are already in original schema
            return naifID
        raise ValueError('Expected extended schema with 8 digits.')
    if naifID[0:2] == '50':
        return '3' + naifID[2:]
    if naifID[0:2] == '20':
        return '2' + naifID[2:]
    raise ValueError('Cannot convert to original schema.')