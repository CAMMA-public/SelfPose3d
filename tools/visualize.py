import cv2
import numpy as np
import pickle
import os
import torch
from tqdm import tqdm
from matplotlib import colors
from copy import deepcopy
from munkres import Munkres
from vedo import Points, Lines, Picture, show, Mesh, Point, Light, Plane, Arrows
import colorcet
import _init_paths
import subprocess
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, get_scale
from utils.transforms import affine_transform_pts_cuda as do_transform
import utils.cameras as cameras
from utils.vis import COLORS, JOINTS_DEF, LIMBS15, LIMBS15_COLORS

PRED_FILE = "./results_publ/cam5_evaluation/predictions_dump_interval3.pkl"

PRED_MESHES_DIR = (
    "./results/adaptOR3D/demo_out/run_07_frozen_ssv_rootnet_syn_posessv"
)
PRED_MESHES_DIR = "./results/adaptOR3D/run_14_1_train_backbone_frozen_rootnet_syn_posessv_pseudo_hrnet_soft/meshes_all"
# PRED_MESHES_DIR = "./results/adaptOR3D/panoptic_multiperson_poseresnet50_prn64_cpn80x80x20_960x512_cam5"

GT_MESHES_DIR = "./results/adaptOR3D/demo_out/gt_val"

# RENDER_DIR = "./results/adaptOR3D/run_07_frozen_ssv_rootnet_syn_posessv/vis_results/set5"
RENDER_DIR = "./results/adaptOR3D/run_14_1_train_backbone_frozen_rootnet_syn_posessv_pseudo_hrnet_soft/vis_results"
RENDER_DIR = "./results/adaptOR3D/panoptic_multiperson_poseresnet50_prn64_cpn80x80x20_960x512_cam5/vis_results_3d"
RENDER_DIR = "./results/adaptOR3D/run_14_1_train_backbone_frozen_rootnet_syn_posessv_pseudo_hrnet_soft/vis_seqs/v3"

KEYNAMES = [
    "160906_ian5_00_03_00001077",
    "160906_ian5_00_03_00001665",
    "160906_ian5_00_03_00002841",
    "160906_band4_00_03_00000329",
    "160906_pizza1_00_03_00000424",
    "160906_pizza1_00_03_00006592",
    "160906_pizza1_00_03_00006400",
    "160422_haggling1_00_03_00000749",
    "160422_haggling1_00_03_00001553",
    "160422_haggling1_00_03_00001733",
    "160422_haggling1_00_03_00001745",
    "160422_haggling1_00_03_00001829",
]
KEYNAMES = ["160906_pizza1_00_03_00000424", "160906_pizza1_00_03_00006592"]

SEQUENCE_NAMES = [
    "160906_pizza1"
]  # "160906_pizza1", "160422_haggling1", "160906_ian5", "160906_band4"
DRAW_SUBSET_MESHES = False
NUMBER_UPDATE_DRAW = 30
MESH_KEYS = [
    #"160906_pizza1_00_03_00000388",
    #"160906_pizza1_00_03_00000544",
    #"160906_pizza1_00_03_00001948",
    #"160906_pizza1_00_03_00003292",
    #"160906_pizza1_00_03_00005260",
    #"160906_pizza1_00_03_00006688",
    #"160906_band4_00_03_00006353",
    #"160906_band4_00_03_00009989",
    #"160422_haggling1_00_03_00001613",
    #"160422_haggling1_00_03_00011741",
    #"160422_haggling1_00_03_00013181",
    #"160906_ian5_00_03_00002997"
]


IMAGE_SIZE = [960, 512]
MAX_WIDTH_HEIGHT = 1920
NUM_CAMS = 5
VIS_ALL = False
NUM_JOINTS = len(JOINTS_DEF)
VIS_GT = False
RENDER_TO_FILE = True
DRAW_PRED_MESHES = False
DRAW_GT_MESHES = False
KPT3D_THRESHOLD = 0.7
MYCMAP = ["white", "white", "white"]
ALPHAS = np.linspace(0.8, 0.5, num=len(MYCMAP))
VIRTUAL_CAM = {
    "pos": (6000.0, 4000.0, 6000.0),
    "focalPoint": (0.0, 10.0, 1000.0),
    "viewup": (0.0, 0.0, 1.0),
}
FIX_CAM = False
USE_TRACKER = True
AZM_UPDATE = -0.5

AXES_OPTS = dict(
    # xtitle="X",  # latex-style syntax
    # ytitle="Y",
    # ztitle="Z",  # many unicode chars are supported (type: vedo -r fonts)
    textScale=1.0,  # make all text 30% bigger
    xrange=(-4000, 4000),
    yrange=(-4000, 4000),
    zrange=(-200, 2000),
    numberOfDivisions=4,  # approximate number of divisions on longest axis
    axesLineWidth=2,
    gridLineWidth=3,
    # zxGrid=True,  # show zx plane on opposite side of the bounding box
    # yzGrid=True,  # show yz plane on opposite side of the bounding box
    xyPlaneColor="black",
    xyGridColor="white",  # darkgreen line color
    xyAlpha=0.05,  # grid opacityrotateY
    ##xTitlePosition=0.5,  # title fractional positions along axis
    # xTitleJustify="top-center",  # align title wrt to its axis
    # yTitleSize=0.02,
    # yTitleBox=True,
    # yTitleOffset=0.05,
    # yLabelOffset=0.4,
    # yHighlightZero=False,  # draw a line highlighting zero position if in range
    # yHighlightZeroColor="white",
    zLineColor="black",
    zTitleColor="black",
    xLineColor="black",
    xTitleColor="black",
    yLineColor="black",
    yTitleColor="black",
    # zTitleBackfaceColor="v",  # violet color of axis title backface
    labelFont="Quikhand",
    # yLabelSize=0.025,  # size of the numeric labels along Y axis
    # yLabelColor="white",  # color of the numeric labels along Y axis
)


def get_transformed_images(im_paths):
    images_orig = [
        cv2.cvtColor(
            cv2.imread(im, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION),
            cv2.COLOR_BGR2RGB,
        )
        for im in im_paths
    ]
    height, width, _ = images_orig[0].shape
    c = np.array([width / 2.0, height / 2.0])
    s = get_scale((width, height), IMAGE_SIZE)
    r = 0
    trans = get_affine_transform(c, s, r, IMAGE_SIZE)
    images = [
        cv2.warpAffine(
            im,
            trans,
            (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR,
        )
        for im in images_orig
    ]
    return images, trans


def crop_img_cv2(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[
        np.min(y_nonzero) : np.max(y_nonzero),
        np.min(x_nonzero) : np.max(x_nonzero),
    ]


def get_gt_dets(preds, i, trans):
    results = {}
    # ground truth
    results["gt_3d"] = np.array(preds[i * NUM_CAMS]["joints_3d"])
    results["gt_3d_vis"] = np.array(preds[i * NUM_CAMS]["joints_3d_vis"])
    gt_2ds = []
    for ii in range(NUM_CAMS):
        cam = deepcopy(preds[i * NUM_CAMS + ii]["camera"])
        cam["cx"] = torch.as_tensor(cam["cx"])
        cam["cy"] = torch.as_tensor(cam["cy"])
        gt_2d = torch.cat(
            [
                do_transform(
                    torch.clamp(
                        cameras.project_pose(
                            torch.tensor(
                                results["gt_3d"][_ii], dtype=torch.float32
                            ),
                            cam,
                        ),
                        -1.0,
                        MAX_WIDTH_HEIGHT,
                    ),
                    torch.as_tensor(trans, dtype=torch.float),
                )[None]
                for _ii in range(results["gt_3d"].shape[0])
            ]
        ).numpy()
        gt_2ds.append(gt_2d)
    results["gt_2ds"] = gt_2ds

    # detections

    results["dt_3d"] = preds[i * NUM_CAMS]["preds_3d"][..., :3]
    dt_2ds = []
    for ii in range(NUM_CAMS):
        cam = deepcopy(preds[i * NUM_CAMS + ii]["camera"])
        cam["cx"] = torch.as_tensor(cam["cx"])
        cam["cy"] = torch.as_tensor(cam["cy"])
        dt_2d = torch.cat(
            [
                do_transform(
                    torch.clamp(
                        cameras.project_pose(
                            torch.tensor(results["dt_3d"][_ii]), cam
                        ),
                        -1.0,
                        MAX_WIDTH_HEIGHT,
                    ),
                    torch.as_tensor(trans, dtype=torch.float),
                )[None]
                for _ii in range(results["dt_3d"].shape[0])
            ]
        ).numpy()
        dt_2ds.append(dt_2d)
    results["dt_2ds"] = dt_2ds

    return results


def draw2d_keypoints(images, keypoints2d):
    assert len(images) == len(keypoints2d), "number mismatched"

    images_out = []
    for im, kpts in zip(images, keypoints2d):
        im_out = deepcopy(im)
        for n, kpt2d in enumerate(kpts):
            for k in eval("LIMBS{}".format(NUM_JOINTS)):
                pt1 = int(kpt2d[k[0], 0]), int(kpt2d[k[0], 1])
                pt2 = int(kpt2d[k[1], 0]), int(kpt2d[k[1], 1])
                cv2.line(
                    im_out,
                    pt1,
                    pt2,
                    [a * 255 for a in colors.to_rgb(COLORS[int(n % 10)])],
                    4,
                    cv2.LINE_AA,
                )
                cv2.circle(im_out, pt1, 3, (255, 225, 225), 2, cv2.LINE_AA)
                cv2.circle(im_out, pt1, 2, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(im_out, pt2, 3, (255, 225, 225), 2, cv2.LINE_AA)
                cv2.circle(im_out, pt2, 2, (0, 0, 0), 1, cv2.LINE_AA)

        images_out.append(im_out)
    return images_out


def images_to_vedo(images):
    X_ROTATE = [90] * NUM_CAMS
    Y_ROTATE = [0] * NUM_CAMS
    Z_ROTATE = [100, 80, 0, 80, 100]
    X_TRANS = [-2000, -2000, -1000, 2000, 2000]
    Y_TRANS = [-2000, 0, 2000, -2000, 0]
    Z_TRANS = [0] * NUM_CAMS
    SCALE = [1.8] * NUM_CAMS
    vedo_pics = []
    for ii, im in enumerate(images):
        vedo_im = (
            Picture(im)
            .rotate_x(X_ROTATE[ii])
            .rotate_y(Y_ROTATE[ii])
            .rotate_z(Z_ROTATE[ii])
        )
        vedo_im.z(Z_TRANS[ii]).y(Y_TRANS[ii]).x(X_TRANS[ii]).scale(SCALE[ii])
        vedo_pics.append(vedo_im)
    return vedo_pics


def get_vedo_points_and_lines(preds):
    pts_pred, lines_pred = None, None
    for n in range(len(preds)):
        pts_start, pts_end = [], []
        for k in eval("LIMBS{}".format(NUM_JOINTS)):
            pts_start.append(
                (preds[n][k[0], 0], preds[n][k[0], 1], preds[n][k[0], 2])
            )
            pts_end.append(
                (preds[n][k[1], 0], preds[n][k[1], 1], preds[n][k[1], 2])
            )
        if n == 0:
            pts_pred = Points(preds[n][..., :3], c="w", r=5, alpha=0.90)
            lines_pred = Arrows(
                pts_start,
                pts_end,
                c=COLORS[int(n % 10)],
                # lw=6,
                res=200,
                alpha=0.9,
                thickness=6,
            )
        else:
            pts_pred += Points(preds[n][..., :3], c="w", r=5, alpha=0.9)
            lines_pred += Arrows(
                pts_start,
                pts_end,
                c=COLORS[int(n % 10)],
                # lw=6,
                res=200,
                alpha=0.9,
                thickness=6,
            )

    return pts_pred, lines_pred


def get_mesh_objs(key, meshes_keys, meshkey_to_dir, tracks):
    if key in meshes_keys:
        mesh_dir = meshkey_to_dir[key]
        mesh_objs = [
            Mesh(os.path.join(mesh_dir, p)).rotateX(180).scale(1000.0)
            for p in sorted(os.listdir(mesh_dir))
        ]
        if tracks[0] != None:
            mesh_objs = [mesh_objs[i] for i in tracks[0 : len(mesh_objs)]]
        mesh_objs = [
            m.subdivide(3)
            .smooth()
            .computeNormals()
            .c(COLORS[int(n % 10)])
            .lighting("metallic")
            .phong()
            for n, m in enumerate(mesh_objs)
        ]
        # .backFaceCulling().c('white').lighting('glossy').lineWidth(0.1).phong() .subdivide(1) .lighting("glossy")
        # scalers = [man.points()[:, 2] for man in mesh_objs]
        # mesh_objs = [
        #     man.cmap(MYCMAP, scals, alpha=ALPHAS) for scals, man in zip(scalers, mesh_objs)
        # ]
        mesh_vedo = None
        if mesh_objs is not None:
            for n in range(len(mesh_objs)):
                if n == 0:
                    mesh_vedo = mesh_objs[n]
                else:
                    mesh_vedo += mesh_objs[n]
    else:
        mesh_vedo = None

    return mesh_vedo


def get_lights():
    p1 = Point([2000, 0, 1000], c="w")
    p2 = Point([0, 0, 2000], c="w")
    p3 = Point([-1000, -500, 500], c="w")
    p4 = Point([-2000, 2000, 500], c="w")
    l1 = Light(p1, c="w", intensity=1.0)
    l2 = Light(p2, c="w", intensity=1.0)
    l3 = Light(p3, c="w", intensity=1.0)
    l4 = Light(p4, c="w", intensity=1.0)

    return l1, l2, l3, l4


def render_results(
    vedo_pics_preds,
    camera,
    points=None,
    lines=None,
    mesh_objs=None,
    offscreen=True,
    interactive=False,
):
    assert NUM_CAMS == 5, "Number of cams not supported"
    l1, l2, l3, l4 = get_lights()
    plt_obj = show(
        mesh_objs,
        vedo_pics_preds[0],
        vedo_pics_preds[1],
        vedo_pics_preds[2],
        vedo_pics_preds[3],
        vedo_pics_preds[4],
        points,
        lines,
        l1,
        l2,
        l3,
        l4,
        axes=AXES_OPTS,
        bg="white",
        # bg2="bb",
        offscreen=offscreen,
        interactive=interactive,
        zoom=1,
        camera=camera,
        size=(1920, 1080),
    )
    return plt_obj


def write_2d_poses(
    images_preds, key_name, SEQUENCE_NAME, dir_name="preds_2d_poses", resize=True,
):
    for kk, im in enumerate(images_preds):
        dir_name_out = os.path.join(
            RENDER_DIR, SEQUENCE_NAME, dir_name, key_name
        )
        os.makedirs(dir_name_out, exist_ok=True)
        im_out = crop_img_cv2(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if resize:
            scale_percent = 50
            width = int(im_out.shape[1] * scale_percent / 100)
            height = int(im_out.shape[0] * scale_percent / 100)
            dim = (width, height)
            im_out = cv2.resize(im_out, dim, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(dir_name_out, "{}_img.jpg".format(kk)), im_out)


def track_3dposes(preds, SEQUENCE_NAME):
    assert len(SEQUENCE_NAME) > 0, "work only with sequence"
    preds = [p for p in preds if SEQUENCE_NAME in p["key"]]
    num_images = int(len(preds) / NUM_CAMS)
    kpt_3d = [
        preds[i * NUM_CAMS]["preds_3d"][..., :3] for i in range(num_images)
    ]
    keys = [preds[i * NUM_CAMS]["key"] for i in range(num_images)]
    scores = [
        preds[i * NUM_CAMS]["preds_3d"][:, 0, -1] for i in range(num_images)
    ]
    num_per_3d = [
        (preds[i * NUM_CAMS]["preds_3d"][:, 0, -1] > KPT3D_THRESHOLD).sum()
        for i in range(num_images)
    ]
    kpt_3d = [k[0:n] for k, n in zip(kpt_3d, num_per_3d)]
    tracks = [[] for p in range(len(kpt_3d))]
    tracks[0] = [(p, p) for p in list(range(kpt_3d[0].shape[0]))]
    
    for pose_id in range(1, len(kpt_3d)):
        cur_poses = kpt_3d[pose_id]
        if cur_poses.shape[0] == 0:
            tracks[pose_id] = [(None, None)]
            continue
        track = np.array([p[1] for p in tracks[pose_id-1]])
        if track[0] is not None:
            prev_poses = kpt_3d[pose_id - 1][track]
        else:
            tracks[pose_id] = [(p, p) for p in list(range(kpt_3d[pose_id].shape[0]))]
            continue
        cost_matrix = np.zeros((prev_poses.shape[0], cur_poses.shape[0]))
        for i, prev_pose in enumerate(prev_poses):
            for j, cur_pose in enumerate(cur_poses):
                mpjpe = np.mean(
                    np.sqrt(np.sum((prev_pose - cur_pose) ** 2, axis=-1))
                )
                cost_matrix[i, j] = mpjpe
        m = Munkres()
        indexes = m.compute((np.array(cost_matrix)).tolist())
        if cost_matrix.shape[1] > len(indexes):
            cur_tracks = [p[1] for p in indexes]
            new_tracks = list(
                set(list(range(cost_matrix.shape[1]))) - set(cur_tracks)
            )
            for nt in new_tracks:
                indexes.append((-1, nt))
        tracks[pose_id] = indexes

    for u_id in range(num_images):
        track_ids = [p[1] for p in tracks[u_id]]
        preds[u_id * NUM_CAMS]["tracks"] = np.array(track_ids)
        if track_ids[0] is not None:
            preds[u_id * NUM_CAMS]["preds_3d"] = preds[u_id * NUM_CAMS]["preds_3d"][
                ..., :3
            ][np.array(track_ids)]

    return preds


def get_mesh_objs_preds(preds, i, meshes_keys_pred, meshkey_to_dir_pred):
    if DRAW_PRED_MESHES:
        if DRAW_SUBSET_MESHES:
            key = preds[i * NUM_CAMS]["key"]
            if key in MESH_KEYS:
                mesh_objs_preds = get_mesh_objs(
                    key,
                    meshes_keys_pred,
                    meshkey_to_dir_pred,
                    preds[i * NUM_CAMS]["tracks"],
                )
            else:
                mesh_objs_preds = None
        else:
            mesh_objs_preds = get_mesh_objs(
                preds[i * NUM_CAMS]["key"],
                meshes_keys_pred,
                meshkey_to_dir_pred,
                preds[i * NUM_CAMS]["tracks"],
            )
    return mesh_objs_preds


def render_sequence(SEQUENCE_NAME):
    preds = pickle.load(open(PRED_FILE, "rb"))
    if USE_TRACKER:
        preds = track_3dposes(preds, SEQUENCE_NAME)
    num_images = int(len(preds) / NUM_CAMS)
    mesh_objs_preds = None
    if DRAW_PRED_MESHES:
        meshes_dir = [
            x[0] for x in os.walk(os.path.join(PRED_MESHES_DIR, "meshes"))
        ][1:]
        meshes_keys_pred = [os.path.basename(m) for m in meshes_dir]
        meshkey_to_dir_pred = {
            k: v for k, v in zip(meshes_keys_pred, meshes_dir)
        }

    first_cam = True
    counter_img = 0
    for i in tqdm(range(num_images), desc="rendering results"):
        out_dir_name = os.path.join(RENDER_DIR, SEQUENCE_NAME, "preds_3d_poses")
        out_dir_name_mesh = os.path.join(RENDER_DIR, SEQUENCE_NAME, "preds_3d_meshes")

        # if preds[i * NUM_CAMS]["key"] not in KEYNAMES:
        #    continue
        im_paths = [preds[i * NUM_CAMS + _n]["image"] for _n in range(NUM_CAMS)]
        images, trans = get_transformed_images(im_paths)
        results = get_gt_dets(preds, i, trans)
        images_preds = draw2d_keypoints(images, results["dt_2ds"])
        vedo_pics_preds = images_to_vedo(images_preds)
        pts_pred, lines_pred = get_vedo_points_and_lines(results["dt_3d"])
        # write_2d_poses(images_preds, preds[i * NUM_CAMS]["key"], SEQUENCE_NAME)
        # DRAW_SUBSET_MESHES = False
        if RENDER_TO_FILE:
            out_filename_pose = os.path.join(out_dir_name,
                preds[i * NUM_CAMS]["key"] if not DRAW_SUBSET_MESHES else f'{counter_img:06d}',
            )
            os.makedirs(os.path.dirname(out_filename_pose), exist_ok=True)
            plt_obj = render_results(
                vedo_pics_preds,
                camera=VIRTUAL_CAM if first_cam else VIRTUAL_CAM2,
                points=pts_pred,
                lines=lines_pred,
            )
            VIRTUAL_CAM2 = plt_obj.camera
            plt_obj.screenshot(out_filename_pose).close()
            if DRAW_SUBSET_MESHES:
                if DRAW_PRED_MESHES and mesh_objs_preds is not None:
                    for pp in range(NUMBER_UPDATE_DRAW):
                        counter_img += 1
                        out_filename_mesh = os.path.join(out_dir_name, f'{counter_img:06d}')
                        os.makedirs(out_dir_name, exist_ok=True)
                        plt_obj = render_results(
                            vedo_pics_preds,
                            camera=VIRTUAL_CAM if first_cam else VIRTUAL_CAM2,
                            mesh_objs=mesh_objs_preds,
                        )
                        plt_obj.screenshot(out_filename_mesh).close()
                        VIRTUAL_CAM2.Azimuth(-0.8)
            else:
                if DRAW_PRED_MESHES and mesh_objs_preds is not None:                    
                    os.makedirs(out_dir_name_mesh, exist_ok=True)
                    out_filename_mesh = os.path.join(out_dir_name_mesh,  preds[i * NUM_CAMS]["key"])
                    if not os.path.isfile(out_filename_mesh+".png"):                    
                        mesh_objs_preds = get_mesh_objs_preds(
                            preds, i, meshes_keys_pred, meshkey_to_dir_pred
                        )                        
                        os.makedirs(out_dir_name_mesh, exist_ok=True)
                        plt_obj = render_results(
                            vedo_pics_preds,
                            camera=VIRTUAL_CAM if first_cam else VIRTUAL_CAM2,
                            mesh_objs=mesh_objs_preds,
                        )
                        plt_obj.screenshot(out_filename_mesh).close()
        else:
            plt_obj = render_results(
                vedo_pics_preds,
                camera=VIRTUAL_CAM if first_cam else VIRTUAL_CAM2,
                points=pts_pred,
                lines=lines_pred,
                mesh_objs=mesh_objs_preds,
                offscreen=False,
                interactive=True,
            )
            plt_obj.close()

        first_cam = False
        counter_img += 1
        
        if not FIX_CAM:
            VIRTUAL_CAM2.Azimuth(AZM_UPDATE)
            # VIRTUAL_CAM2.Zoom(0.7)

    images_to_video(out_dir_name, os.path.dirname(out_dir_name)+"_pose.mp4", fps=25)
    # images_to_video(out_dir_name_mesh, os.path.dirname(out_dir_name_mesh)+"_mesh.mp4", fps=25)

def images_to_video(img_folder, output_vid_file, fps=30):
    """[convert png images to video using ffmpeg]
    Args:
        img_folder ([str]): [path to images]
        output_vid_file ([str]): [Name of the output video file name]
    """
    command = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-threads",
        "16",
        "-pattern_type",
        "glob",
        "-i",
        f"{img_folder}/*.png",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-v",
        "error",
        output_vid_file,
    ]
    print(f'\nRunning "{" ".join(command)}"')
    subprocess.call(command)
    print("\nVideo generation finished")


def main():
    for SEQUENCE_NAME in SEQUENCE_NAMES:
       print("renderig sequence: ", SEQUENCE_NAME)
       render_sequence(SEQUENCE_NAME)


if __name__ == "__main__":
    main()
