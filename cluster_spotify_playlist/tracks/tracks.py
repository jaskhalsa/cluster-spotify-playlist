from typing import Any, Dict, Generator, List

from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

__all__ = ["Tracks"]

MAX_CHUNK_SIZE = 50


class Tracks:
    __client: Spotify

    def __init__(self, credentials: SpotifyClientCredentials) -> None:
        self.__client = Spotify(auth_manager=credentials)

    def get_tracks_info(self, tracks: List[str]) -> Dict[str, Any]:
        tracks_info = []
        track_chunks = list(self.chunk(tracks, MAX_CHUNK_SIZE))
        for chunk in track_chunks:
            chunk_tracks = self.__client.tracks(chunk)["tracks"]
            audio_features = self.__client.audio_features(chunk)
            for track, features in zip(chunk_tracks, audio_features):
                artists = []
                for artist in track["artists"]:
                    artists.append(artist["name"])
                artists = ", ".join(artists)
                formatted_name = track["name"] + " - " + artists
                formatted_audio_features = {
                    "danceability": features["danceability"],
                    "energy": features["energy"],
                    "key": features["key"],
                    "loudness": features["loudness"],
                    "mode": features["mode"],
                    "speechiness": features["speechiness"],
                    "acousticness": features["acousticness"],
                    "instrumentalness": features["instrumentalness"],
                    "liveness": features["liveness"],
                    "valence": features["valence"],
                    "tempo": features["tempo"],
                }
                track_info = {
                    "name": formatted_name,
                    "audio_features": formatted_audio_features,
                }
                tracks_info.append(track_info)
        return tracks_info

    @staticmethod
    def chunk(lst: List[Any], n: int) -> Generator[Any, None, None]:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]
