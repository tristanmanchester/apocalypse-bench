#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict


LEXICON = {
    "AGR": """
agriculture farming agronomy horticulture gardening crop crops field garden vegetable cereal grain legume pulse root crop tuber orchard seed seedbed seedling germination transplanting nursery propagation cutting grafting pruning pollination flowering fruiting harvest threshing winnowing storage granary silage hay straw mulch compost manure green manure composting vermicompost humus soil topsoil loam clay soil sandy soil silt pH fertility nitrogen phosphorus potassium micronutrient lime ash biochar irrigation furrow drip irrigation flood irrigation watering rainwater drainage ditch terrace contour erosion windbreak shelterbelt rotation crop rotation intercropping companion planting cover crop fallow weed weeding hoe shovel spade fork sickle scythe plough plow harrow tillage no-till pest insect aphid beetle caterpillar fungal disease blight mildew rust nematode rodent trap scarecrow animal husbandry livestock poultry goat sheep cattle rabbit beekeeping hive apiary fodder pasture grazing milk egg slaughter hide leather aquaculture fish pond seed saving landrace cultivar yield drought frost greenhouse cold frame hoop house
""",
    "CHEM": """
chemistry chemical reaction reagent solution solvent solute concentration dilution mixture suspension emulsion precipitate crystallization distillation evaporation condensation sublimation filtration decanting extraction leaching washing neutralization oxidation reduction combustion pyrolysis fermentation saponification hydrolysis ester alcohol ethanol methanol glycerol soap lye sodium hydroxide potassium hydroxide caustic lime quicklime slaked lime calcium carbonate limestone chalk ash potash soda ash sodium carbonate bicarbonate vinegar acetic acid citric acid lactic acid hydrochloric acid sulfuric acid nitric acid acid base alkali salt chloride sulfate nitrate carbonate phosphate ammonia ammonium chlorine hypochlorite bleach peroxide iodine tincture antiseptic disinfectant sterilization pH indicator litmus titration hardness scale soapmaking glassmaking ceramics glaze pigment dye mordant tannin ink glue resin tar pitch charcoal activated carbon carbon graphite coke fuel oil fat wax grease biodiesel explosive black powder saltpeter potassium nitrate sulfur charcoal powder corrosion rust electrolysis battery acid electrolyte galvanic cell plating smelting flux slag ore mineral poison toxicity fumes ventilation fire heat boiling point melting point density specific gravity
""",
    "COMMS": """
communication communications signal signaling radio wireless telegraph telephone transmitter receiver transceiver antenna aerial dipole monopole loop antenna longwire Yagi ground plane counterpoise feedline coaxial cable impedance matching standing wave SWR wavelength frequency band kilohertz megahertz hertz modulation AM FM shortwave medium wave longwave VHF UHF HF Morse Morse code code cipher encryption decryption plaintext key one-time pad semaphore flag heliograph mirror lamp lantern beacon flare smoke signal signal fire whistle horn bell drum messenger courier runner relay relay station repeater packet radio crystal radio detector diode galena coil inductor capacitor variable capacitor tuning resonance oscillator spark gap microphone speaker headphones earpiece amplifier vacuum tube transistor rectifier generator battery power supply grounding earth lightning tower mast pole guy wire insulator wire copper wire enamel wire solder connector switch keyer telegraph key line-of-sight horizon propagation ionosphere skip noise interference static jamming bandwidth channel call sign protocol message checksum redundancy error correction logbook map coordinate grid time signal synchronization bulletin noticeboard printing press type typewriter paper ink courier network
""",
    "ENG": """
engineering mechanical engineering civil engineering structural engineering machine mechanism tool workshop repair maintenance fabrication construction building structure frame beam column truss arch bridge roof wall foundation footing masonry brick stone timber lumber plank joist rafter scaffold ladder rope pulley block and tackle winch hoist lever fulcrum wedge screw inclined plane wheel axle bearing bushing shaft crank cam gear sprocket chain belt clutch brake spring flywheel pump piston pump diaphragm pump centrifugal pump hand pump valve check valve pipe tubing siphon hydraulic pneumatic pressure vacuum seal gasket packing leak flow head friction torque tension compression shear bending buckling load load-bearing safety factor stress strain fatigue fracture joint fastener nail screw bolt rivet peg dowel mortise tenon weld solder brazing forge anvil vise file saw chisel hammer drill lathe mill casting mould mold pattern foundry furnace kiln concrete mortar cement lime mortar adobe cob rammed earth drainage canal aqueduct sluice dam waterwheel windmill millstone grain mill press oil press loom cart wheelbarrow bicycle repairability
""",
    "ENR": """
energy power fuel heat work fire combustion flame stove cookstove rocket stove oven kiln furnace hearth chimney flue draft draught ventilation smoke charcoal wood firewood coppice biomass dung fuel peat coal coke gasifier wood gas producer gas biogas methane anaerobic digestion biodigester alcohol fuel ethanol oil lamp kerosene tallow candle wax wick lantern solar solar thermal solar cooker photovoltaic panel charge controller inverter battery lead-acid lithium nickel-cadmium electrolyte acid hydrometer cell voltage current ampere watt watt-hour resistance circuit wire fuse switch generator alternator dynamo magnet coil turbine waterwheel microhydro hydropower penstock head flow rate windmill wind turbine blade rotor furling mechanical power flywheel belt drive pulley steam boiler pressure vessel steam engine Stirling engine heat engine thermoelectric insulation thermal mass heat exchanger radiator condenser evaporator distillation heat refrigeration evaporative cooling ice charcoal kiln retort fuel efficiency energy storage grid direct current alternating current grounding short circuit overcharge sulfation corrosion fire safety carbon monoxide
""",
    "ETH": """
ethics moral philosophy morality justice fairness equity rights duty obligation responsibility accountability consent informed consent autonomy coercion force violence punishment mercy compassion harm harm reduction triage rationing scarcity allocation priority vulnerability children elderly disabled patient refugee prisoner outsider stranger community settlement governance leadership legitimacy authority democracy council assembly vote consensus representation transparency corruption nepotism favoritism conflict of interest due process rule law norm taboo sanction restitution reparation mediation arbitration negotiation conflict resolution trust reputation promise contract oath confidentiality privacy surveillance security collective action common good public good tragedy of the commons free rider mutual aid reciprocity cooperation competition hoarding theft property ownership stewardship commons land tenure inheritance family kinship gender dignity discrimination exploitation slavery forced labor child labor medical ethics quarantine ethics research ethics teaching ethics deception lying truth propaganda rumor panic morale loyalty betrayal self-defense proportionality necessity last resort emergency powers evacuation sacrifice utilitarianism deontology virtue ethics care ethics
""",
    "MAT": """
materials material science metal metallurgy iron steel cast iron wrought iron copper bronze brass tin zinc lead aluminum ore mineral smelting bloomery blast furnace forge charcoal coke flux limestone slag crucible casting mould mold sand casting lost wax ingot billet bar sheet wire annealing tempering hardening quenching case hardening carburizing normalization forging hammering rolling welding brazing soldering rivet corrosion rust galvanizing patina ceramic pottery clay earthenware stoneware porcelain kiln firing glaze grog brick tile refractory lime cement mortar concrete plaster gypsum adobe cob rammed earth glass silica sand soda potash cullet fiber textile cloth linen cotton wool hemp flax jute spinning weaving loom rope cordage twine leather hide tanning tannin parchment paper pulp cellulose wood timber charcoal resin pitch tar glue adhesive casein gelatin starch rubber latex plastic bakelite oil wax pigment dye ink abrasive sandpaper whetstone grindstone insulation waterproofing composite laminate fatigue fracture hardness toughness brittleness elasticity density
""",
    "MEAS": """
measurement metrology measure unit standard calibration accuracy precision error uncertainty tolerance repeatability reproducibility scale ruler yardstick tape measure caliper vernier micrometer gauge feeler gauge protractor angle level spirit level plumb bob square compass divider straightedge template weighing balance scale pan mass weight density volume liter gallon bucket graduated cylinder measuring cup displacement hydrometer thermometer temperature Celsius Fahrenheit boiling point freezing point barometer pressure manometer vacuum gauge hygrometer humidity rain gauge wind vane anemometer clock sundial pendulum hourglass water clock timekeeping calendar date moon star latitude longitude navigation map survey triangulation baseline bearing azimuth altitude sextant quadrant astrolabe clinometer rangefinder pacing odometer speed velocity flow rate head pressure head current voltage resistance multimeter galvanometer ammeter voltmeter wattmeter pH meter indicator titration concentration dose dosage ratio proportion sample sampling experiment control variable trial replicate randomization observation record logbook table chart graph statistics average median variance threshold reference benchmark
""",
    "MED": """
medicine medical first aid emergency medicine nursing diagnosis symptom sign patient airway breathing circulation pulse respiration blood pressure shock bleeding hemorrhage wound cut laceration puncture abrasion burn scald fracture sprain dislocation splint sling bandage dressing gauze compression tourniquet pressure point suture stitch needle sterile sterilization antiseptic disinfectant iodine alcohol chlorhexidine soap clean water saline boil infection sepsis fever inflammation pus abscess tetanus rabies malaria cholera dysentery diarrhea dehydration oral rehydration electrolyte vomiting nutrition anemia pain analgesic aspirin ibuprofen paracetamol morphine antibiotic penicillin amoxicillin sulfa herbal medicine medicinal plant childbirth midwifery pregnancy labor delivery placenta umbilical cord newborn infant breastfeeding obstetrics postpartum hemorrhage miscarriage contraception anatomy physiology skin bone muscle tendon nerve artery vein heart lung abdomen eye ear dental tooth extraction hygiene quarantine isolation triage evacuation stretcher hypothermia heatstroke poisoning snakebite allergic reaction asthma seizure unconsciousness
""",
    "ORG": """
organisation organization governance administration coordination logistics planning operations management leadership council committee assembly meeting agenda minutes record ledger registry census roster schedule rota shift duty role responsibility accountability delegation chain of command command post incident command work crew labor workforce training apprenticeship supervisor quartermaster storehouse inventory stockpile ration rationing distribution supply procurement salvage repair maintenance priority triage queue allocation warehouse cache map route transport convoy messenger communication bulletin noticeboard report audit inspection checklist standard operating procedure rule policy law dispute mediation arbitration discipline sanction reward morale trust cooperation mutual aid commons resource management water committee food committee health committee security watch patrol guard access control evacuation shelter camp settlement household family vulnerable children elders sick quarantine sanitation burial fire brigade emergency response risk register contingency redundancy succession rotation corruption theft hoarding black market barter currency accounting budget debt credit contract ownership land tenure census taking
""",
    "PED": """
pedagogy teaching education instruction learning training lesson curriculum syllabus course workshop classroom apprentice apprenticeship mentor coach student learner teacher instructor trainer demonstration practice drill exercise assessment test quiz exam oral exam practical exam checklist rubric competency mastery feedback correction review repetition spaced repetition memory recall retrieval practice explanation example analogy diagram model hands-on simulation scenario role-play peer teaching group work individual work lecture discussion Socratic method questioning observation supervision safety briefing procedure protocol skill craft literacy numeracy reading writing arithmetic measurement map reading first aid training hygiene training farming training workshop safety tool use maintenance troubleshooting diagnosis record keeping logbook notebook blackboard slate chalk paper manual handbook field guide reference library oral tradition storytelling mnemonic song chant poster sign label translation language terminology vocabulary prerequisite progression beginner intermediate advanced certification pass fail remediation error misconception curiosity motivation discipline attendance schedule rota classroom management
""",
    "PH": """
public health sanitation hygiene epidemiology outbreak epidemic pandemic infection control disease prevention surveillance case definition contact tracing quarantine isolation vaccination immunization herd immunity vector control mosquito fly flea lice rat rodent water drinking water potable well spring river pond rainwater cistern tank reservoir filtration sand filter slow sand filter charcoal filter boiling chlorination chlorine hypochlorite bleach iodine turbidity fecal contamination coliform sewage latrine toilet pit latrine composting toilet urine feces sludge wastewater greywater drainage stagnant water handwashing soap ash laundry bathing food safety food storage spoilage refrigeration pasteurization cooking cross-contamination nutrition malnutrition vitamin protein calorie breastfeeding infant feeding shelter crowding ventilation smoke indoor air carbon monoxide waste disposal garbage burial carcass pest control camp settlement clinic triage health worker community health health education risk communication morbidity mortality fever diarrhea cholera dysentery typhoid malaria measles tuberculosis respiratory infection wound infection parasites worms deworming mental health trauma stress clean birth maternal health
""",
    "SAFE": """
safety hazard risk danger accident injury prevention mitigation emergency rescue evacuation shelter fire safety fire flame smoke carbon monoxide ventilation explosion pressure pressure vessel boiler steam burn scald electrical safety shock short circuit grounding fuse battery acid chemical safety poison toxic fumes gas chlorine ammonia acid caustic lye solvent methanol contamination infection biohazard sharp blade knife axe saw chisel hammer drill workshop personal protective equipment glove goggles mask respirator apron boots helmet harness rope knot fall ladder scaffold roof trench collapse cave-in confined space drowning flood river ice hypothermia heat exhaustion heatstroke dehydration cold exposure animal bite snakebite insect sting allergy first aid bleeding tourniquet splint burn dressing alarm watch patrol guard perimeter lighting lock barricade crowd control panic violence weapon self-defense storage labeling segregation childproofing food safety spoilage mold water safety sanitation quarantine procedure checklist stop-work inspection maintenance redundancy fail-safe safety factor warning sign training supervision incident report near miss
""",
}


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_query(text):
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/search")
    parser.add_argument("--limits", default="1000,5000,10000")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    limits = [int(item) for item in args.limits.split(",") if item.strip()]
    all_by_limit = {limit: set() for limit in limits}
    article_categories = {limit: defaultdict(set) for limit in limits}
    output = {
        "url": args.url,
        "limits": limits,
        "categories": {},
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for category, lexicon in LEXICON.items():
        query = normalize_query(lexicon)
        output["categories"][category] = {"queryTerms": len(query.split()), "runs": {}}
        for limit in limits:
            started = time.time()
            try:
                response = post_json(args.url, {"query": query, "limit": limit}, args.timeout)
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"{category} limit={limit} failed: {error.code} {body}") from error
            hits = response.get("hits", [])
            article_ids = {hit["article_id"] for hit in hits if hit.get("article_id")}
            for article_id in article_ids:
                all_by_limit[limit].add(article_id)
                article_categories[limit][article_id].add(category)
            output["categories"][category]["runs"][str(limit)] = {
                "chunks": len(hits),
                "uniqueArticles": len(article_ids),
                "latencyMs": response.get("latencyMs"),
                "wallMs": round((time.time() - started) * 1000),
                "topTitles": [
                    {
                        "title": hit.get("title"),
                        "article_id": hit.get("article_id"),
                        "score": hit.get("score"),
                    }
                    for hit in hits[:10]
                ],
            }
            print(
                f"{category} limit={limit} chunks={len(hits)} articles={len(article_ids)} latencyMs={response.get('latencyMs')}",
                file=sys.stderr,
                flush=True,
            )

    output["union"] = {}
    for limit in limits:
        overlap_counts = defaultdict(int)
        for categories in article_categories[limit].values():
            overlap_counts[len(categories)] += 1
        output["union"][str(limit)] = {
            "uniqueArticles": len(all_by_limit[limit]),
            "overlapBuckets": dict(sorted(overlap_counts.items())),
        }

    output["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
