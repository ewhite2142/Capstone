import googlemaps

gmaps = googlemaps.Client(key='AIzaSyA5hijNL12eH_MHMIKdLicir449saeO6c0')

loc = gmaps.reverse_geocode((46.8529, -113.929))

print("loc:\n{}".format(loc))
loc
loc[0]['address_components']
