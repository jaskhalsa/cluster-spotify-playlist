from typing import List

from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

__all__ = ["Playlist"]


class Playlist:
    __client: Spotify

    def __init__(self, credentials: SpotifyClientCredentials) -> None:
        self.__client = Spotify(auth_manager=credentials)

    def get_tracks_for_playlist(self, playlist_id: str) -> List[str]:
        playlist = self.__client.playlist(playlist_id)["tracks"]
        tracks = []
        while playlist:
            for item in playlist["items"]:
                tracks.append(item["track"]["uri"])
            if playlist["next"]:
                playlist = self.__client.next(playlist)
            else:
                playlist = None
        return tracks
