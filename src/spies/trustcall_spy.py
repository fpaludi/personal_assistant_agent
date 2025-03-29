import logging

logger = logging.getLogger(__name__)


class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

    def extract_tool_info(self, schema_name: str = "Memory"):
        """Extract information from tool calls for both patches and new memories.

        Args:
            schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
        """

        # Initialize list of changes
        changes = []

        for call_group in self.called_tools:
            for call in call_group:
                if call['name'] == 'PatchDoc':
                    logger.info(f"Call: {call}")
                    if call['args']['patches'][0]['op'] == "remove":
                        changes.append({
                            'type': 'remove',
                            'doc_id': call['args']['json_doc_id'],
                            'planned_edits': call['args']['planned_edits'],
                        })
                    else:
                        changes.append({
                            'type': 'update',
                            'doc_id': call['args']['json_doc_id'],
                            'planned_edits': call['args']['planned_edits'],
                            'value': call['args']['patches'][0]['value']
                        })
                elif call['name'] == schema_name:
                    changes.append({
                        'type': 'new',
                        'value': call['args']
                    })

        # Format results as a single string
        result_parts = []
        for change in changes:
            if change['type'] == 'update':
                result_parts.append(
                    f"Document {change['doc_id']} updated:\n"
                    f"Plan: {change['planned_edits']}\n"
                    f"Added content: {change['value']}"
                )
            else:
                result_parts.append(
                    f"New {schema_name} created:\n"
                    f"Content: {change['value']}"
                )

        return "\n\n".join(result_parts)
