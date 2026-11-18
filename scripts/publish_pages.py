"""Publish to the existing Pages configuration without changing repository settings."""
import argparse
import json
import os
import subprocess
import time


def api(endpoint, method='GET'):
    result = subprocess.run(['gh', 'api', '--method', method, endpoint], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def inspect(repo):
    site = api(f'repos/{repo}/pages')
    mode = site.get('build_type', 'legacy')
    if mode not in ('legacy', 'workflow'):
        raise ValueError('Unknown GitHub Pages build type')
    if mode == 'legacy' and site.get('source') != {'branch': 'main', 'path': '/docs'}:
        raise ValueError('Existing Pages source must be main:/docs, or use GitHub Actions; settings were not changed')
    return mode, site['html_url']


def publish_legacy(repo, commit, tries=60, pause=time.sleep):
    mode, url = inspect(repo)
    if mode != 'legacy':
        raise ValueError('Use actions/deploy-pages for workflow mode')
    api(f'repos/{repo}/pages/builds', 'POST')
    for _ in range(tries):
        build = api(f'repos/{repo}/pages/builds/latest')
        if build.get('commit') == commit:
            if build.get('status') == 'built':
                return url
            if build.get('status') == 'errored':
                raise RuntimeError(str(build.get('error')))
        pause(5)
    raise TimeoutError('Could not verify Pages build for the accepted model commit')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inspect-only', action='store_true')
    p.add_argument('--commit')
    a = p.parse_args()
    repo = os.environ['GITHUB_REPOSITORY']
    if a.inspect_only:
        mode, url = inspect(repo)
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f'build_type={mode}\nurl={url}\n')
        print(f'Pages mode: {mode}')
    else:
        if not a.commit:
            p.error('--commit is required')
        print('Published:', publish_legacy(repo, a.commit))
