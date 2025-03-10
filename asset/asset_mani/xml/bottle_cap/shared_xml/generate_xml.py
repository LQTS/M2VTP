import os
import xml.etree.ElementTree as ET
import trimesh
import numpy as np
import shutil
from scipy.spatial.transform import Rotation as R

def main():
    template_path = 'bottle_cap_template.xml'
    decomposed_mesh_dir = '/home/zjunesc/LQT_PhD_WorkSpace/meshdata'
    types = [type for type in os.listdir(decomposed_mesh_dir) if 'core-bottle' in type]
    failed = os.listdir('../failed')
    cnt = 0
    for tp in types[:]:
        if (tp+'.xml') in failed:
            continue
        # open template xml
        tree = ET.parse(template_path)
        root = tree.getroot()
        asset = ET.SubElement(root, 'asset')
        hand_body, obj_body, obj_body1 = [body for body in root.iter('body')]
        obj_geom = obj_body.findall('geom')[0]
        obj_body.remove(obj_geom)
        obj_geom1 = obj_body1.findall('geom')[0]
        obj_body1.remove(obj_geom1)

        path = os.path.join(decomposed_mesh_dir, tp, 'coacd')
        # get all decomposed meshes
        pieces = [piece for piece in os.listdir(path) if 'coacd_convex_piece' in piece]
        if len(pieces)<=1:
            continue
        peices_mesh = [trimesh.load_mesh(os.path.join(path, piece)) for piece in pieces]
        # rotate
        trans = np.eye(4)
        trans[:-1, :-1] = R.from_euler('x', 90, degrees=True).as_matrix()
        for id, mesh in enumerate(peices_mesh):
            mesh.apply_transform(trans)
            mesh.export(os.path.join('../assets', tp + '-' + pieces[id]))
        peices_mesh_z_max = [np.max(mesh.vertices[:, -1]) for mesh in peices_mesh]
        peices_mesh_z_min = [np.min(mesh.vertices[:, -1]) for mesh in peices_mesh]

        # get top piece
        top_piece = pieces[np.argmax(peices_mesh_z_max)]
        pieces.remove(top_piece)
        # get min height
        min_height = np.min(peices_mesh_z_min)
        # add top pieces
        add_piece('cap' , asset, obj_body, path, top_piece, tp, '0.3 0.6 0.6 1', manipulated=True)
        for piece in pieces:
            add_piece(piece[:-4], asset, obj_body1, path, piece, tp, '0.6 0.2 0.2 1')
        obj_body.attrib['pos'] = f'0 0 {-min_height/10}'
        obj_body1.attrib['pos'] = f'0 0 {-min_height/10}'

        hand_body.attrib['pos'] = f'-0.35 -0.03 {np.max(peices_mesh_z_max-min_height)/10+0.07}'  #-0.37

        saved_path = f'../{tp}.xml'
        print(saved_path)
        tree.write(saved_path, 'UTF-8')
        cnt += 1
    print(f'total envs is {cnt}')


def add_piece(name, asset, obj_body, path, piece, type, rgb='0.68 0.68 0.68 1', manipulated=False):

    obj_name = type + '-' + piece
    mesh_attrib = {'name': obj_name[:-4], 'file': obj_name, 'scale': '0.1 0.1 0.1'}
    ET.SubElement(asset, 'mesh', mesh_attrib)
    if manipulated:
        geom_attrib = {'name': name,
                       'mesh': obj_name[:-4],
                       'mass': '0.01',
                       # 'density': '1240',
                       'pos': '0 0 0',
                       'rgba': rgb,
                       'type': 'mesh',
                       'condim': "6",
                       'priority': "1",
                       'friction': "0.5 0.01 0.003"
                       }
        ET.SubElement(obj_body, 'geom', geom_attrib)
    else:
        geom_attrib = {'name': name,
                       'mesh': obj_name[:-4],
                       'mass': '0.01',
                       # 'density': '1240',
                       'pos': '0 0 0',
                       'rgba': rgb,
                       'type': 'mesh',
                       'contype': "2",
                       'conaffinity': "2"
                       }
        ET.SubElement(obj_body, 'geom', geom_attrib)
    # shutil.copy(os.path.join(path, piece), os.path.join('assets', obj_name))


if __name__ == '__main__':
    main()

