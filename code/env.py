"""
    Environment with one object at center
        external: one robot, one camera
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, SceneConfig, OptifuserConfig, ArticulationJointType
from sapien.core.pysapien import ActorBase, VulkanRenderer
from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
from utils import process_angle_limit, get_random_number
# import trimesh
import ipdb
import math

class ContactError(Exception):
    pass

class SVDError(Exception):
    pass

class Env(object):
    def __init__(self, flog=None, show_gui=True, render_rate=20, timestep=1/500,
                 object_position_offset=(0.0, 0.0, 0.0), succ_ratio=0.1, set_ground=False,
                 static_friction=0.30, dynamic_friction=0.30, gravity=[0, 0, -9.8]):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.object_position_offset = object_position_offset
        self.have_table  =False

        # engine and renderer
        self.engine = sapien.Engine(0, 0.001, 0.005)
        
        render_config = OptifuserConfig()
        render_config.shadow_map_size = 8192
        render_config.shadow_frustum_size = 10
        render_config.use_shadow = False
        render_config.use_ao = True

        self.renderer = sapien.OptifuserRenderer(config=render_config)
        # self.renderer = sapien.VulkanRenderer()
        self.renderer.enable_global_axes(False)
        
        self.engine.set_renderer(self.renderer)

        # GUI
        self.window = False
        if show_gui:
            self.renderer_distance = 2.5
            self.renderer_controller = sapien.OptifuserController(self.renderer)
            self.renderer_controller.set_camera_position(self.renderer_distance, self.renderer_distance, self.renderer_distance)
            self.renderer_controller.set_camera_rotation(-np.pi *3/ 4, -np.arcsin(1/np.sqrt(3)))
            # The principle axis of the camera is the x-axis
            # viewer.set_camera_xyz(x=-4, y=0, z=2)
            # # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
            # # The camera now looks at the origin
            # viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)

        # scene
        scene_config = SceneConfig()
        scene_config.gravity = gravity
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0
        scene_config.default_static_friction = static_friction
        scene_config.default_dynamic_friction = dynamic_friction

        self.scene = self.engine.create_scene(config=scene_config)

        # ground
        ground_material = self.engine.create_physical_material(0.3, 0.3, 0.01)
        self.scene.add_ground(altitude=0, render=True, material=ground_material)

        # box-wall
        # create_box(self.engine, self.scene, Pose([0, -0.25, 0.15]), [0.05, 0.25, 0.15], (0.8, 0.3, 0.2), 100.0, 0.01, 1000, 'box_w')
        # box-leaning
        # create_box(self.engine, self.scene, Pose([0.25, 0.25, 0.15]), [0.05, 0.25, 0.15], (0.8, 0.4, 0.3), 100.0, 0.01, 1000, 'box_l1')
        # thickness = 0.01
        # length = 0.3*2+0.1
        # theta = np.pi/6
        # create_box(self.engine, self.scene, Pose([0.2-0.3/np.tan(theta)+(length/2)*np.cos(theta)-(thickness/2)*np.sin(theta), 0.4, (length/2)*np.sin(theta)+(thickness/2)*np.cos(theta)], [np.cos(theta), 0, np.sin(theta), 0]), [thickness*0.5, 0.2, length*0.5], (0.8, 0.4, 0.3), 100.0, 0.01, 1000, 'box_l2')

        # # box_gap
        # create_box(self.engine, self.scene, Pose([-0.7, 0.25, -0.5]), [0.1, 0.25, 0.5], (0.8, 0.5, 0.5), 100.0, 0.01, 1000, 'box_g')

        if show_gui:
            self.renderer_controller.set_current_scene(self.scene)

        self.scene.set_timestep(timestep)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_point_light([1, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1])
        
        # default Nones
        self.object = None
        self.object_target_joint = None

        # check contact
        self.check_contact = False
        self.contact_error = False
        self.raise_contact_error  =False

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.renderer_controller.set_camera_position(x, y, z)
        self.renderer_controller.set_camera_rotation(yaw, pitch)
        self.renderer_controller.render()

    def load_object(self, urdf, material, state='closed', scale=1.0, density=1.0, stiffness=0.1, damping=10, lieDown=False, down = False, given_joint_angles=None, given_pose=None, fix_root_link=False, rand_orientation=0.1):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = fix_root_link
        loader.scale = scale
        self.object = loader.load(urdf, {"material": material, "density": density})
        if given_pose:
            pose = given_pose
        else:
            if not lieDown and not down:
                pose = Pose([self.object_position_offset[0], self.object_position_offset[1], self.object_position_offset[2]], [1,0,0,0]+np.random.randn(4)*rand_orientation) # Quaternion
            elif lieDown:
                # pose = Pose([self.object_position_offset[0], self.object_position_offset[1], self.object_position_offset[2]], [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0])
                pose = Pose([self.object_position_offset[0], self.object_position_offset[1], self.object_position_offset[2]], [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0]+np.random.randn(4)*rand_orientation)
            elif down:
                pose = Pose([self.object_position_offset[0], self.object_position_offset[1], self.object_position_offset[2]], [0, 0, 1, 0]+np.random.randn(4)*rand_orientation)
        self.object.set_root_pose(pose)
        # print('object pose: ', self.object.get_root_pose())

        # compute link actor information
        self.all_link_ids = [l.get_id() for l in self.object.get_links()]
        self.all_joint_types = [j.type for j in self.object.get_joints()]
        self.movable_link_ids = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                self.movable_link_ids.append(j.get_child_link().get_id())
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=stiffness, damping=damping)

        # set initial qpos
        joint_angles = []
        self.joint_angles_lower = []
        self.joint_angles_upper = []
        # for l in self.object.get_links():
        #     print(l.mass)

        for j in self.object.get_joints():
            # print(j.get_limits())
            if j.get_dof() == 1:
                # print(j.type)
                l = process_angle_limit(j.get_limits()[0, 0], j.type)
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1], j.type)
                # print("l, r:", l, r)
                self.joint_angles_upper.append(float(r))
                if state == 'closed':
                    joint_angles.append(float(l))
                elif state == 'open':
                    joint_angles.append(float(r))
                elif state == 'middle':
                    joint_angles.append(float((l + r) / 2))
                elif state == 'random-middle':
                    joint_angles.append(float(get_random_number(l, r)))
                elif state == 'random-closed-middle':
                    if np.random.random() < 0.5:
                        joint_angles.append(float(get_random_number(l, r)))
                    else:
                        joint_angles.append(float(l))
                else:
                    raise ValueError('ERROR: object init state %s unknown!' % state)
                # j.set_limits(np.array([[max(joint_angles[-1]-0.0001, l), min(joint_angles[-1]+0.0001, r)]]))
                # print(j.get_limits())

        if given_joint_angles:
            joint_angles = given_joint_angles
        # print('joint_angles: ', joint_angles)
        self.object.set_qpos(joint_angles)
        self.object.set_drive_target(joint_angles)
        # print("joint_angles = ", joint_angles)
        # print(self.object.get_drive_target())
        # print(self.object.get_qpos())
        # print(self.object.get_pose())
        # print(self.object.get_root_pose())
        # print(self.object.get_qvel())
        # print(self.object.get_qlimits())
        # self.object.set_qlimits(np.array([self.joint_angles_lower, self.joint_angles_upper]))
        # self.object.set_qpos([-0.6, -0.6])
        return joint_angles

    def create_box(self, pose: sapien.Pose, half_size, color=(0.5, 0.5, 0.7), friction=10.0, restitution=0.01, density = 1000, name='box') -> sapien.Actor:
        half_size = np.array(half_size)
        builder: sapien.ActorBuilder = self.scene.create_actor_builder()
        box_pose = sapien.Pose([0., 0., 0.])
        box_material = self.engine.create_physical_material(friction, friction, restitution)
        builder.add_box_shape(pose=box_pose, size=half_size, material=box_material, density=density)  # Add collision shape
        builder.add_box_visual(size=half_size, color=color, pose=box_pose)  # Add visual shape
        box: sapien.Actor = builder.build_static(name=name)
        # Or you can set_name after building the actor
        # box.set_name(name)
        box.set_pose(pose)
        return box

    def create_table(self, pose: sapien.Pose, size_x, size_y, height, thickness=0.1, color=(0.8, 0.6, 0.4), friction=1.0, restitution=0.01, density = 1000, name='table') -> sapien.Actor:
        """Create a table (a collection of collision and visual shapes)."""
        builder = self.scene.create_actor_builder()
        
        # Tabletop
        tabletop_pose = sapien.Pose([0., 0., -thickness / 2+height])  # Make the top surface's z equal to 0
        tabletop_half_size = [size_x / 2, size_y / 2, thickness / 2]
        table_material = self.engine.create_physical_material(friction, friction, restitution)
        builder.add_box_visual(pose=tabletop_pose, size=tabletop_half_size, color=color)
        builder.add_box_shape(pose=tabletop_pose, size=tabletop_half_size, material=table_material, density=density)
        
        # Table legs (x4)
        for i in [-1, 1]:
            for j in [-1, 1]:
                x = i * 0.9 * (size_x - thickness) / 2
                y = j * 0.9 * (size_y - thickness) / 2
                table_leg_pose = sapien.Pose([x, y, -thickness/2+height / 2])
                table_leg_half_size = [thickness / 2, thickness / 2, (height-thickness) / 2]
                builder.add_box_shape(pose=table_leg_pose, size=table_leg_half_size, density=density, material=table_material)
                builder.add_box_visual(pose=table_leg_pose, size=table_leg_half_size, color=color)

        table = builder.build_static(name=name)
        table.set_pose(pose)
        self.have_table = True
        return table

    def create_slope(self, urdf, pose: sapien.Pose, scale=1.0, friction=1.0, restitution=0.01, density = 1000) -> sapien.Actor:
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = scale
        material = self.engine.create_physical_material(friction, friction, restitution)
        object = loader.load(urdf, {"material": material, "density": density})
        object.set_root_pose(pose)
        return object
    
    # def create_groove(self, urdf, pose: sapien.Pose, scale=1.0, friction=1.0, restitution=0.01, density = 1000) -> sapien.Actor:
    #     loader = self.scene.create_urdf_loader()
    #     loader.fix_root_link = True
    #     loader.scale = scale
    #     material = self.engine.create_physical_material(friction, friction, restitution)
    #     self.object = loader.load(urdf, {"material": material, "density": density})
    #     self.object.set_root_pose(pose)
    #     return self.object
    
    def create_groove(self, gap=0.15, height=0.1, friction=1.0, restitution=0.01, density = 1000, color=(0.2, 0.1, 0.7)) -> sapien.Actor:
        builder = self.scene.create_actor_builder()
        
        # Tabletop
        left_pose = sapien.Pose([-1.0 - gap/2, 0., height/2])  # Make the top 
        right_pose = sapien.Pose([1.0 + gap/2, 0., height/2])  # Make the top surface's z equal to 0
        half_size = [1.0, 1.0, height/2]
        groove_material = self.engine.create_physical_material(friction, friction, restitution)
        builder.add_box_visual(pose=left_pose, size=half_size, color=color)
        builder.add_box_shape(pose=left_pose, size=half_size, material=groove_material, density=density)
        builder.add_box_visual(pose=right_pose, size=half_size, color=color)
        builder.add_box_shape(pose=right_pose, size=half_size, material=groove_material, density=density)
        

        groove = builder.build_static(name='groove')
        groove.set_pose(Pose([0,0,0], [1,0,0,0]))
        return groove

    def create_multiple_scenes(self, friction=1.0, restitution=0.01, density = 1000) -> sapien.Actor:
        builder = self.scene.create_actor_builder()
        table = self.create_table(Pose([0, 0, 0]), 1.5, 1.5, 1.0, 0.1, (0.8, 0.6, 0.4), 1.0, 0.01, 1000, 'table')
        slope = self.create_slope(r'../../Shape_data/env_assets/slope/urdf/slope.urdf', Pose([0, 0.75, 1.], [0.5,-0.5,0.5,-0.5]), scale=3.0)
        groove_material = self.engine.create_physical_material(friction, friction, restitution)
        builder.add_box_visual(pose=Pose([0,-0.755,0.955], [1,0,0,0]), size=[0.75,0.005,0.045], color=(0.8, 0.6, 0.4))
        builder.add_box_shape(pose=Pose([0,-0.755,0.955], [1,0,0,0]), size=[0.75,0.005,0.045], material=groove_material, density=density)
        builder.add_box_visual(pose=Pose([0,-0.835,0.915], [1,0,0,0]), size=[0.75,0.075,0.005], color=(0.8, 0.6, 0.4))
        builder.add_box_shape(pose=Pose([0,-0.835,0.915], [1,0,0,0]), size=[0.75,0.075,0.005], material=groove_material, density=density)
        builder.add_box_visual(pose=Pose([0,-0.915,0.955], [1,0,0,0]), size=[0.75,0.005,0.045], color=(0.8, 0.6, 0.4))
        builder.add_box_shape(pose=Pose([0,-0.915,0.955], [1,0,0,0]), size=[0.75,0.005,0.045], material=groove_material, density=density)
        groove = builder.build_static(name='groove')
        groove.set_pose(Pose([0,0,0], [1,0,0,0]))
        builder_w = self.scene.create_actor_builder()
        builder_w.add_box_visual(pose=Pose([-0.8,0,1.14], [1,0,0,0]), size=[0.05,0.75,0.15], color=(0.8, 0.6, 0.4))
        builder_w.add_box_shape(pose=Pose([-0.8,0,1.14], [1,0,0,0]), size=[0.05,0.75,0.15], material=groove_material, density=density)
        wall = builder_w.build_static(name='wall')
        wall.set_pose(Pose([0,0,0], [1,0,0,0]))
        return table, slope, groove, wall

    def get_target_part_axes(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[1, 0]), float(mat[2, 0]), float(-mat[0, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')
        return joint_axes

    def get_target_part_axes_new(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[0, 0]), float(-mat[1, 0]), float(mat[2, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')

        return joint_axes


    def set_target_object_part_actor_id2(self, actor_id):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id     # not movable
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))

        # get the link handler
        for j in self.object.get_joints():
            if j.get_child_link().get_id() == actor_id:
                self.target_object_part_actor_link = j.get_child_link()

        # monitor the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_child_link().get_id() == actor_id:
                self.target_object_part_joint_id = idx
                self.target_object_part_joint_type = j.type
            idx += 1


    def get_object_qpos(self):
        return self.object.get_qpos()

    def get_object_root_pose(self):
        return self.object.get_root_pose()

    def get_target_part_qpos(self):
        qpos = self.object.get_qpos()
        # ipdb.set_trace()
        return float(qpos[self.target_object_part_joint_id])
    
    def get_target_part_pose(self):
        return self.target_object_part_actor_link.get_pose()

    def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict, raise_error = False):
        self.check_contact = True
        self.check_time = 0
        self.check_contact_strict = strict
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids
        self.contact_error = False
        self.raise_contact_error = raise_error

    def end_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict, raise_error = False):
        self.check_contact = False
        self.check_contact_strict = strict
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids
        self.raise_contact_error = raise_error

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def render(self):
        if self.show_gui and (not self.window):
            self.window = True
            self.renderer_controller.show_window()
        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.renderer_controller.render()

    def step(self):
        self.current_step += 1
        # print('stepping')
        self.scene.step()
        # print('stepped')
        if self.check_contact:
            # print('checking contact')
            self.check_contact_is_valid()
            if not self.check_contact_is_valid() and self.raise_contact_error:
                # print('raise contact error now')
                raise ContactError()
                # pass

    # check the first contact: only gripper links can touch the target object part link
        # check the first contact: only gripper links can touch the target object part link
    def check_contact_is_valid(self, if_pre = False):
        self.contacts = self.scene.get_contacts()
        contact = False; valid = False
        self.check_time += 1
        # print("check_time: ", self.check_time)
        for c in self.contacts:
            aid1 = c.actor1.get_id()
            aid2 = c.actor2.get_id()
            has_impulse = False
            for p in c.points:
                if abs(p.impulse @ p.impulse) > 1e-4:
                    has_impulse = True
                    break
            if has_impulse:
                if (aid1 in self.robot_gripper_actor_ids and aid2 == self.target_object_part_actor_id) or \
                   (aid2 in self.robot_gripper_actor_ids and aid1 == self.target_object_part_actor_id):
                       contact, valid = True, True
                # if (aid1 in self.robot_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
                #    (aid2 in self.robot_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
                #     print('contact ERROR: gripper and non-target part')
                #     if self.check_contact_strict:
                #         self.contact_error = True
                #         return False
                #     else:
                #         contact, valid = True, True
                if (aid1 in self.robot_gripper_actor_ids and aid2 not in self.all_link_ids) or \
                    (aid2 in self.robot_gripper_actor_ids and aid1 not in self.all_link_ids):
                    print(f'contact ERROR: gripper and other: {c.actor1.get_name()}, {c.actor2.get_name()}')
                    if self.check_contact_strict and not if_pre:
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, False
                if (aid1 == self.robot_hand_actor_id and aid2 not in self.all_link_ids) or \
                    (aid2 == self.robot_hand_actor_id and aid1 not in self.all_link_ids):
                    print(f'contact ERROR: robot hand and other: {c.actor1.get_name()}, {c.actor2.get_name()}')
                    self.contact_error = True
                        # print(aid1, aid2)
                        # print(self.all_link_ids, self.robot_gripper_actor_ids, self.robot_hand_actor_id)
                        # for p in c.points:
                        #     print(p.position, p.impulse)
                if (aid1 == self.robot_hand_actor_id and aid2 in self.all_link_ids) or \
                    (aid2 == self.robot_hand_actor_id and aid1 in self.all_link_ids):
                    print(f'contact ERROR: robot hand and object: {c.actor1.get_name()}, {c.actor2.get_name()}')
                    if self.check_contact_strict:
                        self.contact_error = True
                    else:
                        contact, valid = True, False
                # starting pose should have no collision at all
                if (aid1 in self.robot_gripper_actor_ids or aid1 == self.robot_hand_actor_id or \
                    aid2 in self.robot_gripper_actor_ids or aid2 == self.robot_hand_actor_id) and self.check_time == 1:
                        print(f'contact ERROR: initial state interaction: {c.actor1.get_name()}, {c.actor2.get_name()}')
                        self.contact_error = True
                        return False

        if contact and valid:
            self.check_contact = False
        return True



    def close_render(self):
        if self.window:
            self.renderer_controller.hide_window()
        self.window = False
    
    def wait_to_start(self):
        print('press q to start\n')
        while not self.renderer_controller.should_quit:
            self.scene.update_render()
            if self.show_gui:
                self.renderer_controller.render()

    def close(self):
        if self.show_gui:
            self.renderer_controller.set_current_scene(None)
        self.scene = None
