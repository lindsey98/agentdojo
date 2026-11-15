import io
import logging
import unittest

from rich.console import Console
from rich.logging import RichHandler
from agentdojo.logging import OutputLogger


class MarkupLoggingTests(unittest.TestCase):
    def test_terminal_and_errors_keep_untrusted_markup_literal(self):
        payload = '[INFORMATION BLOCK] payload [/INFORMATION BLOCK] [bold]literal[/bold]'
        message = {'role': 'tool', 'content': payload, 'error': None}
        output = io.StringIO()
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            root.handlers = [RichHandler(console=Console(file=output, width=2000), markup=True)]
            root.setLevel(logging.INFO)
            logger = OutputLogger(None)
            logger.log([message], suite_name='slack', pipeline_name='gemma+progent', user_task_id='user_task_12')
            logger.log_error(payload)
            message_error = {'role': 'tool', 'content': '', 'error': payload}
            logger.log([message, message_error], suite_name='slack', user_task_id='user_task_12')
        finally:
            root.handlers, root.level = handlers, level
        self.assertIn('[/INFORMATION BLOCK]', output.getvalue())
        self.assertIn('[bold]literal[/bold]', output.getvalue())
        self.assertEqual(message['content'], payload)
        self.assertEqual(message_error['error'], payload)

    def test_live_table_keeps_payload_literal(self):
        output = io.StringIO()
        class Live:
            def update(self, table):
                Console(file=output, width=2000).print(table)
        payload = '[/INFORMATION BLOCK] [bold]literal[/bold]'
        logger = OutputLogger(None, live=Live())
        logger.log([{'role': 'tool', 'content': payload, 'error': None}], suite_name='slack', user_task_id='user_task_12')
        self.assertIn(payload, output.getvalue())
