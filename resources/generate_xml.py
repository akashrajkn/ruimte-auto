import os


def generate_xml():
    # will generate an xml file per track
    # 5 dec 2017; this leads to segmentation faults
    # sometimes there will be a segmentation fault on a track that would run fine before or will after

    rootdir = '/home/bram/Documents/CI/torcs-server/torcs-1.3.7-patched/data/tracks/'

    # for root, subdirs, files in os.walk(rootdir):

    # all_files = [x[0] for x in os.walk(directory)]
    tracks = ['a-speedway', 'aalborg', 'dirt-1', 'dirt-2', 'dirt-3', 'dirt-4', 'dirt-5', 'dirt-6', 'e-track-1', 'e-track-4', 'e-track-5', 'eroad',
              'g-track-1', 'mixed-1', 'mixed-2', 'b-speedway', 'c-speedway',  'd-speedway', 'e-speedway', 'f-speedway',  'g-speedway', 'michigan',
              'alpine-1', 'alpine-2', 'brondehach', 'corkscrew', 'e-track-2', 'e-track-3', 'e-track-6', 'forza', 'g-track-2', 'g-track-3',
              'ole-road-1', 'ruudskogen', 'spring', 'street-1', 'wheel-1', 'wheel-2']

    xml_data = ''

    oval_tracks = ['michigan', 'a-speedway', 'b-speedway', 'c-speedway', 'd-speedway', 'e-speedway', 'e-track-5' 'g-speedway',  'f-speedway']
    dirt_tracks = ['dirt-1', 'dirt-2', 'dirt-3', 'dirt-4', 'dirt-5', 'dirt-6', 'mixed-1', 'mixed-2']

    with open('track.txt', 'r') as f:
        xml_data = f.read()

    for track in tracks:
        category = 'road'
        if track in oval_tracks:
            category = 'oval'
        elif track in dirt_tracks:
            category = 'dirt'

        new_xml = xml_data.format(track, category)

        with open('xmls/' + track + '.xml', 'w+') as f:
            f.write(new_xml)



def multiple_tracks():
    # generates one xml that specifies to use all tracks
    # so far we have only seen it run a lap in 35.11 seconds everytime
    # whether this means no updates or very slow updates we do not know

    tracks = ['a-speedway', 'aalborg', 'dirt-1', 'dirt-2', 'dirt-3', 'dirt-4', 'dirt-5', 'dirt-6', 'e-track-1', 'e-track-4', 'e-track-5', 'eroad',
              'g-track-1', 'mixed-1', 'mixed-2', 'b-speedway', 'c-speedway',  'd-speedway', 'e-speedway', 'f-speedway',  'g-speedway', 'michigan',
              'alpine-1', 'alpine-2', 'brondehach', 'corkscrew', 'e-track-2', 'e-track-3', 'e-track-6', 'forza', 'g-track-2', 'g-track-3',
              'ole-road-1', 'ruudskogen', 'spring', 'street-1', 'wheel-1', 'wheel-2']


    oval_tracks = ['michigan', 'a-speedway', 'b-speedway', 'c-speedway', 'd-speedway', 'e-speedway', 'e-track-5' 'g-speedway',  'f-speedway']
    dirt_tracks = ['dirt-1', 'dirt-2', 'dirt-3', 'dirt-4', 'dirt-5', 'dirt-6', 'mixed-1', 'mixed-2']

    xml_data = ''

    with open('track.txt', 'r') as f:
        xml_data = f.read()

    replacement_text = '<!--Add tracks here-->'

    new_text = ''

    for i, track in enumerate(tracks):
        category = 'road'
        if track in oval_tracks:
            category = 'oval'
        elif track in dirt_tracks:
            category = 'dirt'

        new_text += '    <section name="{0}">\n      <attstr name="name" val="{1}"/>\n      <attstr name="category" val="{2}"/>\n    </section>\n'.format(str(i+1), track, category)


    xml_data = xml_data.replace(replacement_text, new_text)

    with open('all_tracks.xml', 'w+') as f:
        f.write(xml_data)



if __name__ == '__main__':
    generate_xml()
    # multiple_tracks()
