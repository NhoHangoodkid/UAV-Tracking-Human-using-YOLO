"""Helpers for displaying tracker IDs without cumulative ID inflation."""


class ActiveTrackIdMapper:
    """Map raw tracker IDs to compact IDs for currently active tracks."""

    def __init__(self):
        self._raw_to_display = {}
        self._display_to_raw = {}
        self.seen_raw_ids = set()

    def update(self, raw_ids):
        """Return compact display IDs for the raw IDs present in one frame."""
        current_raw_ids = list(dict.fromkeys(raw_ids))
        current_raw_set = set(current_raw_ids)

        for raw_id in list(self._raw_to_display):
            if raw_id not in current_raw_set:
                display_id = self._raw_to_display.pop(raw_id)
                self._display_to_raw.pop(display_id, None)

        available_display_ids = iter(
            display_id
            for display_id in range(
                1, len(current_raw_ids) + len(self._display_to_raw) + 1
            )
            if display_id not in self._display_to_raw
        )

        display_ids = []
        for raw_id in current_raw_ids:
            display_id = self._raw_to_display.get(raw_id)
            if display_id is None:
                display_id = next(available_display_ids)
                self._raw_to_display[raw_id] = display_id
                self._display_to_raw[display_id] = raw_id
            self.seen_raw_ids.add(raw_id)
            display_ids.append(display_id)

        return display_ids

    @property
    def active_count(self):
        """Return the number of raw tracks currently mapped to display IDs."""
        return len(self._raw_to_display)
