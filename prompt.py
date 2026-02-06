import time
import spacy

import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.ops import unary_union
from geopy.geocoders import Nominatim
from shapely.geometry import LineString

from geodesk import *


def visualize_street(merged_gdf, title="Street geometry"):
    if merged_gdf is None or merged_gdf.empty:
        print("No geometry to plot.")
        return

    ax = merged_gdf.plot(figsize=(8, 8), color="red", linewidth=2)
    ax.set_title(title)
    plt.show()


def find_street(geolocator, streets, name, city):
    # Step 1: Geocode city + street
    start = time.time()
    loc = geolocator.geocode(f"{name}, {city}, Belgium", exactly_one=True)
    if not loc:
        return None
    print(f"Geocode time: {time.time() - start:.3f}s")

    # Step 2: Initial bbox (just to seed the search)
    if hasattr(loc, "raw") and "boundingbox" in loc.raw:
        bb = loc.raw["boundingbox"]
        bbox = (float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1]))
    else:
        buffer = 0.02
        bbox = (loc.longitude - buffer, loc.latitude - buffer,
                loc.longitude + buffer, loc.latitude + buffer)

    streets_local = streets.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    streets_local = streets_local[streets_local["name"] == name]

    if streets_local.empty:
        print(f"No street '{name}' found in {city}")
        return None

    # Step 3: Get ALL road segments with same name (nationwide)
    all_segments = streets[streets["name"] == name]

    # Step 4: Build graph for all those segments
    G = nx.Graph()
    for idx, row in all_segments.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            start_node, end_node = tuple(coords[0]), tuple(coords[-1])
            G.add_edge(start_node, end_node, geom=geom)

    # Step 5: Traverse graph starting from seed segment(s)
    connected_lines = []
    visited_edges = set()

    def dfs(node):
        for nbr in G.neighbors(node):
            edge = tuple(sorted((node, nbr)))
            if edge not in visited_edges:
                visited_edges.add(edge)
                geom = G[node][nbr]["geom"]
                connected_lines.append(geom)
                dfs(nbr)

    # Start from each endpoint of the seed segment(s)
    for _, row in streets_local.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            for node in (tuple(geom.coords[0]), tuple(geom.coords[-1])):
                dfs(node)

    # Step 6: Merge geometries
    if not connected_lines:
        return None

    merged_geom = unary_union(connected_lines)
    merged = gpd.GeoDataFrame(geometry=[merged_geom], crs=streets.crs)
    return merged


def combine_find():
    geolocator = Nominatim(user_agent="my_geo_app")
    streets = gpd.read_file("belgium-250910-free.shp/gis_osm_roads_free_1.shp")
    municipalities = gpd.read_file("belgium-250910-free.shp/gis_osm_places_free_1.shp")

    # Example list of locations
    locations = [
        # ["Lammerstraat", "Ghent"],
        # ["Korenmarkt", "Ghent"],
        # ["Hoogpoort", "Ghent"],
        # ["Botermarkt", "Ghent"],
        # ["Heuvelstraat", "Tielt-Winge"],
        # ["Hellekens", "Tielt-Winge"],
        ["Domeinstraat", "Kessel-Lo"]
    ]

    for l in locations:
        print(l)
        result = find_street(geolocator, streets, l[0], l[1])
        visualize_street(result, title=f"{l[0]}, {l[1]}")
        print(result.geometry)
        time.sleep(1)


def nlp():
    nlp = spacy.load("en_core_web_sm")  # or "xx_ent_wiki_sm" for multilingual
    text = """Onze regio is donderdagochtend getroffen door een onweer. Hierdoor is er in en rond Leuven wateroverlast. Wees voorzichtig als je door het water rijdt. Op de Brusselsesteenweg staan ter hoogte van garage Ceulemans een aantal wagens in panne, omdat ze het water trotseerden.

    De berichten van straten en tunnels die onder water staan, stromen binnen. De tunnel onder het station van Leuven staat blank, ook de Eenmeilaan is ondergelopen. Ook Bertem is zwaar getroffen en ook de afrit van de E40 staat onder water. Daarnaast opende de Delhaize in Kessel-Lo later dan normaal door de wateroverlast. Als je door het water rijdt, doe dat dan zo langzaam mogelijk. 

    Weerman Bram Verbruggen gaf ons meer informatie: “Aan ons meetpunt in Oud-Heverlee is er zeer veel water gevallen. Zo’n 65 liter per vierkante meter, dat valt normaal gezien op drie weken tijd. Dat zorgt bijna onoverkomelijk voor lokale problemen. In Oost-Brabant is minder gevallen. Op het meetpunt aan mijn huis in Langdorp, Aarschot, is in totaal 16 liter gevallen. Het was wel duidelijk dat vannacht tussen 5 en 6 uur de grote piek was.” 

    De burgemeester van Leuven, Mohamed Ridouani (Vooruit) communiceerde ook over de situatie: “Het hevige onweer van deze ochtend veroorzaakt ook in Leuven wateroverlast. De brandweer, politie en stadsdiensten zijn volop in de weer om de grootste problemen zo snel mogelijk te verhelpen. We volgen de situatie nauw op. Er is wateroverlast in meerdere straten, fietspaden en tunnels, zoals de Eenmeilaan en omgeving, het Stoemperspad met de fietstunnel onder de Tiensesteenweg en in de Oosterweeltunnel onder het Martelarenplein.”

    “Het onweer van donderdagochtend dat gepaard ging met bakken regen op zeer korte tijd, zorgde ook in Leuven voor heel wat overlast. Op verschillende plaatsen konden de riolen het vele water niet slikken. Een aantal riooldeksels werden door de druk van het water losgeslagen wat voor gevaarlijke situaties zorgde”, zegt Leuvens politiewoordvoerder Marc Vranckx.

    “In de omgeving van het provinciaal domein in Kessel-Lo kwamen verschillende straten blank te staan. Onder meer een deel van de Eenmeilaan moest worden afgesloten omdat verkeer er niet meer mogelijk was. Ook de Domeinstraat stond helemaal onder water net zoals een deel van de Gemeentestraat. Het water trok slechts langzaam weg. Het provinciaal domein zelf kon niet op het voorziene uur worden geopend. Ook de winkel Delhaize langs de Diestsesteenweg in Kessel-Lo had last van binnenstromend water en ging later open”, weet Vranckx.

    “De Oostertunnel aan het station moest worden afgesloten voor alle verkeer omdat er een diepe plas water stond. Het verkeer werd over het Martelarenplein geleid. Ook op de Sint-Jansbergsesteenweg zorgde een diepe plas voor verkeersproblemen. Een dame kwam midden in de watermassa vast te zitten in haar wagen en werd bevrijd door de brandweer.  In de Bondgenotenlaan kwam een deel van het dak van een gebouw naar beneden. De bewoners haastten zich de straat op, maar er vielen gelukkig geen gewonden. De brandweer ging ter plaatse om tijdelijke maatregelen te nemen”, aldus nog Vranckx.

    “De Wipstraat, Herfstlaan, Lentedreef en de Domeinstraat zijn ook afgesloten”, zo vult Marc Vranckx, woordvoerder van de Leuvense politie, aan. “Ook de Aarschotsesteenweg ter hoogte van de brug aan de E314 heeft te kampen met water op de rijbaan, doorgaand verkeer is nog mogelijk. In de sporthal in de Rijschoolstraat is er water in de sporthal gelopen. In de Parijsstraat ontstond een zinkgat door een waterlek. De weg is afgesloten van Drie Engelenberg tot Sint Barbarastraat”, aldus nog Vranckx.

    In het UZ Gasthuisberg werden enkele operatiezalen buiten gebruik gesteld en is parking West tijdelijk onbereikbaar.

    Verder meldt ook de bibliotheek van Bertem dat ze tijdelijk sluiten tot en met 11 augustus door de wateroverlast. Ze verwijzen de bibliotheekgebruikers door naar de uitleenpost van Leefdaal, Vlieg-uit 6. In de Parijsstraat in Leuven is er ook een zinkgat ontstaan waardoor er geen voertuigen meer kunnen passeren.

    Daarnaast blijft ook het Provinciedomein in Kessel-Lo voorlopig gesloten voor publiek. Wanneer het terug zal openen kunnen ze nu nog niet inschatten, klinkt het. 

    Ook uitvaartzorg Cromboom in Leefdaal werd getroffen. Het dak van de winkel is ingestort waardoor deze tijdelijk buiten gebruik is. Ze blijven wel open en zorgen voor een kleine winkelruimte in een ander deel van het gebouw.
    """

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:   # FAC = facility, often streets
            print(ent.text, ent.label_)
# combine_find()
