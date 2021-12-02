import sys

filter_domain = {'music.release', 'authority.musicbrainz', '22-rdf-syntax-ns#type', 'book.isbn',
                 'common.licensed_object', 'tv.tv_series_episode', 'type.namespace', 'type.content',
                 'type.permission', 'type.object.key', 'type.object.permission', 'type.type.instance',
                 'topic_equivalent_webpage', 'dataworld.freeq',
                 'soft.isbn', 'base.wordnet', 'base.mtgbase', 'base.yupgrade'}
filter_rel = {'type.object.type', 'kg.object_profile.prominent_type',
              'music.recording.artist', 'common.topic.description', 'en', 'freebase.valuenotation.has_value',
              'common.topic.article', 'music.performance_role.track_performances',
              'people.profession.people_with_this_profession',
              'base.schemastaging.nutrition_information.per_quantity',
              'location.statistical_region.unemployment_rate', 'authority.openlibrary.book',
              'music.track_contribution.contributor', 'tv.tv_program.episodes', 'music.recording.length',
              'freebase.valuenotation.is_reviewed', 'music.composition.recordings', 'music.recording.song',
              'book.book_edition.book', 'book.book.editions', 'music.recording.contributions',
              'common.webpage.topic', 'common.topic.webpage', 'common.image.appears_in_topic_gallery',
              'medicine.drug_label_section.subject_drug', 'medicine.manufactured_drug_form.label_sections',
              'medicine.drug_label_section.prominent_warning', 'organization.organization_relationship.as_of_date',
              'authority.imdb.name', 'base.schemastaging.nutrition_information.nutrient_amount',
              'music.recording.tracks', 'book.book_edition.place_of_publication', 'music.album.artist',
              'measurement_unit.dated_integer.source', 'freebase.valuenotation.has_no_value',
              'media_common.cataloged_instance.stanford_opac', 'authority.stanford.control',
              'people.place_lived.person', 'music.recording.releases',
              'film.film_regional_release_date.film_release_region', 'common.topic.official_website',
              'award.award_nomination.year', 'common.topic.topical_webpage'}
special_rel = {'common.topic.notable_types', 'type.object.name'}

seed_set = set()
with open('webqsp/seed.txt') as wqsp, open('CWQ/seed.txt') as cwq:
    for line in wqsp:
        seed_set.update([line.strip()])
    for line in cwq:
        seed_set.update([line.strip()])
print("seed size: %d" % len(seed_set))

input = "data/fb_en.txt"
output = "manual_fb_filter.txt"
f_in = open(input)
f_out = open(output, "w")
num_line = 0
num_reserve = 0
for line in f_in:
    splitline = line.strip().split("\t")
    num_line += 1
    if len(splitline) < 3:
        continue
    rel = splitline[1]

    flag = False
    for domain in filter_domain:
        if domain in rel:
            flag = True
            break
    if flag:
        continue
    elif rel in filter_rel:
        continue
    elif rel in special_rel:
        if not (splitline[0] in seed_set or splitline[2] in seed_set):
            continue

    f_out.write(line)
    num_reserve += 1
    if num_line % 1000000 == 0:
        print(num_line, num_reserve)
f_in.close()
f_out.close()
print(num_line, num_reserve)
