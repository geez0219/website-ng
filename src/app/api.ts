export interface API {
    name: string;
    displayName: string;
    github_link: string;
    type: string;
    children?: API[];
}
