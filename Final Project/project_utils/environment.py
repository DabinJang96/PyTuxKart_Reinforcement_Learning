import pystk
import numpy as np
import matplotlib.pyplot as plt

class PyTuxActionCritic:
  _singleton = None
  RESCUE_TIMEOUT = 30

  def __init__(self, screen_width=128, screen_height=96, steps=1000, verbose=False):
      assert PyTuxActionCritic._singleton is None, "Cannot create more than one pytux object"
      PyTuxActionCritic._singleton = self
      self.config = pystk.GraphicsConfig.hd()
      self.config.screen_width = screen_width
      self.config.screen_height = screen_height
      pystk.init(self.config)
      self.k = None
      self.t = 0
      self.state = None
      self.track = None
      self.last_rescue = 0
      self.distance = 0
      self.steps = steps
      self.fig = None
      self.ax = None
      self.verbose = verbose
      self.last_frame = None
      if verbose:
            self.fig, self.ax = plt.subplots(1, 1)

  @staticmethod
  def _point_on_track(distance, track, offset=0.0):
      """
      Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
      Returns a 3d coordinate
      """
      node_idx = np.searchsorted(track.path_distance[..., 1],
                                  distance % track.path_distance[-1, 1]) % len(track.path_nodes)
      d = track.path_distance[node_idx]
      x = track.path_nodes[node_idx]
      t = (distance + offset - d[0]) / (d[1] - d[0])
      return x[1] * t + x[0] * (1 - t)

  @staticmethod
  def _to_image(x, proj, view):
      p = proj @ view @ np.array(list(x) + [1])
      return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

  def restart(self, track):
    self.state = pystk.WorldState()
    self.track = pystk.Track()

    self.last_rescue = 0
    self.t = 0
    self.distance = 0

    if self.k is not None and self.k.config.track == track:
      self.k.restart()
      self.k.step()
    else:
      if self.k is not None:
          self.k.stop()
          del self.k
      config = pystk.RaceConfig(num_kart=1, laps=1,track=track)
      config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

      self.k = pystk.Race(config)
      self.k.start()
      self.k.step()

    # self.state = pystk.WorldState()
    # self.track = pystk.Track()

    self.last_frame = np.array(self.k.render_data[0].image)

    return np.array(self.k.render_data[0].image)

  def getState(self):
    if (self.k is not None):
      yield np.array(self.k.render_data[0].image)

    yield np.zeros((self.config.screen_height, 
                     self.config.screen_width, 3))
    
  def step(self, action, verbose=False):
    """
    Play a level (track) for a single round.
    :param track: Name of the track
    :param controller: low-level controller, see controller.py
    :param max_frames: Maximum number of frames to play for
    :param verbose: Should we use matplotlib to show the agent drive?
    :return: state, reward, done, time
    """
    # action.brake = False

    reward = 0
  
    self.state.update()
    self.track.update()

    kart = self.state.players[0].kart

    im = self.k.render_data[0].image

    proj = np.array(self.state.players[0].camera.projection).T
    view = np.array(self.state.players[0].camera.view).T
    WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
    aim_point_world = self._point_on_track(kart.distance_down_track, self.track)
    # aim_point_image = self._to_image(aim_point_world, proj, view)
    # loc = self._to_image(kart.location, proj, view)
    # print(" ", np.linalg.norm([kart.location, aim_point_world]))
    # print(" ", WH2*(1+self._to_image(kart.location, proj, view)))

    # current_distance = 0.1 * kart.location[0] + 0.9 * kart.location[-1]
    # current_distance = max(kart.location[0], kart.location[-1])

    current_distance = kart.distance_down_track

    # if (current_distance > self.distance):
    #   reward += 0.01
    # else:
    #   reward -= 1

    # print(" ", self.track.path_nodes)
    # print(np.arccos(np.array(im).flatten() ))


    if np.isclose(kart.overall_distance / self.track.length, 1.0, atol=2e-3):
        if verbose:
            print("Finished at t=%d" % self.t)
            reward = self.steps - self.t
        return np.array(im), reward, True, current_distance # reward for finish

    if (self.t == self.steps):
      reward = -200
      return np.array(im), reward, True, current_distance

    current_vel = np.linalg.norm(kart.velocity)

    if current_vel < 1.0 and self.t - self.last_rescue > PyTuxActionCritic.RESCUE_TIMEOUT:
        self.last_rescue = self.t
        action.rescue = True
        reward -= 200
        if (current_distance > 1000):
          current_distance = 0
        return np.array(im), np.floor(np.sign(current_distance) * np.square(current_distance)) + reward, True, current_distance

    if self.verbose:
      self.ax.clear()
      # self.ax.add_artist(plt.Circle(WH2*(1+self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
      # self.ax.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
      self.ax.imshow(im)                
      plt.pause(1e-3)

    self.k.step(action)
    self.t += 1

    # if (current_vel < 20):
    #   reward -= 0.1

    reward -= 0.02 * np.linalg.norm([kart.location, aim_point_world])

    self.distance = max(self.distance, current_distance)
    # self.last_frame = np.array(im)

    return np.array(im), reward, False, current_distance# penalty for each additional step

  def close(self):
    """
    Call this function, once you're done with PyTux
    """
    if self.k is not None:
        self.k.stop()
        del self.k
    pystk.clean()