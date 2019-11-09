import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http'; 
import { Router, NavigationEnd } from '@angular/router';
import { Subscription } from 'rxjs';

import { NestedTreeControl } from '@angular/cdk/tree';
import { MatTreeNestedDataSource } from '@angular/material/tree';

import { API } from '../api';

@Component({
  selector: 'app-api',
  templateUrl: './api.component.html',
  styleUrls: ['./api.component.css']
})
export class ApiComponent implements OnInit {
  apiList: API[];
  selectedAPI: string;
  currentSelection: string;
  currentAPIText: string;
  routerSubscription: Subscription;

  treeControl: NestedTreeControl<API>;
  dataSource: MatTreeNestedDataSource<API>;

  constructor(private http: HttpClient,
    private router: Router,) { }

  ngOnInit() {
    this.treeControl = new NestedTreeControl<API>(node => node.children);
    this.dataSource = new MatTreeNestedDataSource<API>();

    this.getAPIStructure();

    this.routerSubscription = this.router.events.subscribe((e: any) => {
      if (e instanceof NavigationEnd) {
        this.parseURL();
      }
    });
  }

  hasChild = (_: number, node: API) => !!node.children && node.children.length > 0;

  flatten(arr) {
    var ret: API[] = [];
    for (let a of arr) {
      if (a.children) {
        ret = ret.concat(this.flatten(a.children));
      } else {
        ret = ret.concat(a);
      }
    }

    return ret;
  }

  getAPIStructure() {
    this.http.get('assets/api/structure.json', {responseType: 'text'}).subscribe(data => {
      this.apiList = <API[]>JSON.parse(data);
      
      this.dataSource.data = this.apiList;
      this.treeControl.dataNodes = this.apiList;
      
      this.parseURL();
    });
  }

  private parseURL() {
    var pathComponents = this.router.url.split("/api/");
    var name = "";
    if (pathComponents.length > 1) {
      name = pathComponents[1];
    }
    else {
      name = "";
    }
    if (name === "") {
      this.updateCurrentAPI(this.apiList[0].children[0]);

      this.treeControl.expand(this.treeControl.dataNodes[0]);
    }
    else {
      var a: API[] = this.flatten(this.apiList).filter(api => {
        var split: string[] = api.name.split("/");
        var matchName: string[] = (name + ".md").split("/");
        return split[split.length - 1] === matchName[matchName.length - 1];
      });

      this.updateAPIContent(a[0]);
      // this.expandNodes(a[0], name);
    }
  }

  expandNodes(api: API, apiName: string) {
    console.log(apiName);
    for (let node of this.treeControl.dataNodes) {
      console.log(node);
    }
    // this.treeControl.expand(this.treeControl.dataNodes[5]);
    // this.treeControl.expand(this.treeControl.dataNodes[5].children[0]);
  }

  updateCurrentAPI(api: API) {
    var path = this.updateAPIContent(api);

    this.router.navigateByUrl('/api/' + path);
  }

  private updateAPIContent(api: API) {
    window.scroll(0, 0);
    this.selectedAPI = api.name;
    this.currentSelection = 'assets/api/' + api.name;
    var path = api.name.substring(0, api.name.length - 3);
    if (!path.startsWith("fe")) {
      path = "fe/" + path;
    }
    this.getSelectedAPIText();
    return path;
  }

  getSelectedAPIText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentAPIText = data;
    });
  }

}
